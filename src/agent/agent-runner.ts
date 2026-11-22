/**
 * AgentRunner
 *
 * Agent SDK의 query()를 SSE 스트림으로 감싸 실행합니다.
 * - runStream(): SSE 이벤트를 yield하는 AsyncGenerator
 * - 도구 호출마다 trace 이벤트를 실시간 emit
 * - sessionId 재사용으로 채팅형 연속 대화 지원
 * - 취소: sessionStore.cancelRequest() → 내부 agentGen.return() → 스트림 종료
 */

import {
  query,
  tool,
  createSdkMcpServer,
  type HookCallback,
} from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";
import { env } from "../config/env.js";
import type {
  SessionStore,
  SessionState,
  ClarificationQuestion,
  TraceEntry,
} from "./session-store.js";

// ── 공개 타입 ──────────────────────────────────────────────

export interface ProcessRequest {
  question: string;
  memberId: string;
  sessionId?: string;
  answers?: Record<string, string>;
  requestId: string;
}

export interface Reference {
  locator: string;
  description: string;
}

/** SSE 이벤트 유니온 타입 */
export type SseEvent =
  | { event: "connected"; data: { sessionId: string } }
  | { event: "token"; data: { text: string } }
  | { event: "trace"; data: TraceEntry }
  | {
      event: "clarification";
      data: { questions: ClarificationQuestion[]; type?: "LOGIC_REVIEW" | "ERROR_CAUSE" };
    }
  | {
      event: "done";
      data: {
        answer: string;
        references: Reference[];
        type?: "LOGIC_REVIEW" | "ERROR_CAUSE";
        usage: { elapsedMs: number; tokens?: number; filesRead?: number; dbQueries?: number };
      };
    }
  | { event: "error"; data: { sessionId: string; error: string } };

// ── 시스템 프롬프트 ────────────────────────────────────────

function buildSystemPrompt(memberId: string): string {
  return `당신은 회원사(${memberId})의 ERP 시스템을 분석하는 전문 에이전트입니다.

[필수 초기 작업]
분석을 시작하기 전에 반드시 set_query_type 도구를 호출해 질의 유형을 선언하세요.
  - LOGIC_REVIEW: 특정 로직/기능이 어떻게 구현되어 있는지 확인하는 질문
  - ERROR_CAUSE: 계산값 오류, 0원, 미반영 등 원인을 찾는 질문

[분석 절차]
1. set_query_type으로 유형 선언
2. 질의가 모호하거나 추가 정보가 필요하면 request_clarification을 호출하고 분석을 중단하세요
3. ERP Git / DB 도구로 근거를 찾아 답변하세요

[탐색 효율 규칙 — 반드시 준수]
- 병렬 호출: 서로 독립적인 검색은 한 턴에 여러 도구를 동시에 호출하세요 (예: searchCode 두 개를 한 번에 호출)
- 중복 금지: 이미 readFileRange로 읽은 파일은 절대 다시 읽지 마세요. 대화 기록에 내용이 있으면 그대로 재사용하세요
- 범위 확보: readFileRange 호출 시 lineStart~lineEnd를 충분히 넓게 잡아 한 번에 필요한 로직을 모두 읽으세요
- 추적 순서: ERP 로직은 JSP → Controller → Service → Mapper XML 순서로 단계별 추적하세요
- 탐색 한도: 최대 ${env.claudeMaxTurns}턴. 각 턴에서 가능한 한 많은 정보를 병렬로 수집하세요

[도구 사용 원칙]
- 코드 분석: searchCode 또는 listFiles 로 탐색 → readFileRange 로 내용 확인
- 데이터 분석: describeTable → executeQuery 순서로 진행
- 모든 주장은 파일 경로:줄번호 또는 테이블명으로 근거를 제시하세요
- 추측 금지. 소스/DB 근거가 없으면 명확히 밝히세요

[답변 형식]
- 핵심 결론을 먼저, 세부 근거를 뒤에
- 코드 참조: 파일경로:줄번호 형식
- DB 참조: 테이블명 또는 컬럼명 명시`;
}

// ── Windows 호환 명령어 ────────────────────────────────────

function buildMcpCommand(scriptPath: string): { command: string; args: string[] } {
  const isWin = process.platform === "win32";
  return isWin
    ? { command: "cmd", args: ["/c", "npx", "tsx", scriptPath] }
    : { command: "npx", args: ["tsx", scriptPath] };
}

// ── AgentRunner ───────────────────────────────────────────

export class AgentRunner {
  constructor(private readonly sessionStore: SessionStore) {}

  /**
   * 질의를 처리하고 SSE 이벤트를 실시간으로 yield합니다.
   *
   * 이벤트 순서:
   *   connected → trace (반복) → clarification | done | error
   *
   * 채팅 연속 대화:
   *   sessionId를 재사용하면 Agent SDK의 resume으로 이전 대화 맥락을 이어갑니다.
   *   answers 없이 question만 보내면 follow-up 질문으로 처리됩니다.
   */
  async *runStream(req: ProcessRequest): AsyncGenerator<SseEvent> {
    const startMs = Date.now();

    // ── 중복 요청 차단 ────────────────────────────────────
    if (this.sessionStore.isRequestRunning(req.requestId)) {
      yield {
        event: "error",
        data: { sessionId: req.sessionId ?? "", error: "동일한 requestId로 이미 처리 중인 요청이 있습니다." },
      };
      return;
    }

    // ── 세션 관리 ─────────────────────────────────────────
    let session: SessionState | undefined;
    let sessionId: string;

    if (req.sessionId) {
      session = this.sessionStore.getSession(req.sessionId);
      if (!session) {
        yield {
          event: "error",
          data: { sessionId: req.sessionId, error: `세션을 찾을 수 없습니다: ${req.sessionId}` },
        };
        return;
      }
      sessionId = req.sessionId;
      // done 상태 세션에 follow-up이 오면 재활성화
      if (session.status === "done") {
        this.sessionStore.updateSession(sessionId, { status: "active" });
      }
    } else {
      sessionId = this.sessionStore.createSession(req.memberId, req.question);
    }

    // ── 연결 확인 이벤트 ──────────────────────────────────
    yield { event: "connected", data: { sessionId } };

    // ── 이벤트 큐 (백그라운드 → 스트림 브릿지) ──────────
    const queue: Array<SseEvent | null> = [];
    let waitResolve: (() => void) | null = null;
    let cancelled = false;

    const push = (event: SseEvent | null) => {
      if (cancelled) return;
      queue.push(event);
      waitResolve?.();
      waitResolve = null;
    };

    // ── 추적 상태 ─────────────────────────────────────────
    let pendingClarification: ClarificationQuestion[] | null = null;
    let queryType: "LOGIC_REVIEW" | "ERROR_CAUSE" | undefined = session?.queryType;
    const references: Reference[] = [];
    let filesRead = 0;
    let dbQueries = 0;
    let totalTokens = 0;
    /** 현재까지 완료된 LLM 턴 수 (message_start 기준) */
    let turnCount = 0;
    /** 현재까지 완료된 도구 호출 횟수 */
    let toolCallCount = 0;

    // ── 커스텀 인프로세스 MCP 도구 ───────────────────────
    const setQueryTypeTool = tool(
      "set_query_type",
      "분석 시작 시 질의 유형을 선언합니다. LOGIC_REVIEW: 구현 확인, ERROR_CAUSE: 오류 원인 분석",
      { type: z.enum(["LOGIC_REVIEW", "ERROR_CAUSE"]) },
      async (args) => {
        queryType = args.type;
        push({
          event: "trace",
          data: {
            label: "유형 판별",
            detail: args.type === "LOGIC_REVIEW" ? "로직 문의" : "오류 원인 확인",
            elapsedMs: Date.now() - startMs,
          },
        });
        return { content: [{ type: "text" as const, text: `질의 유형: ${args.type}` }] };
      }
    );

    const requestClarificationTool = tool(
      "request_clarification",
      "질의가 모호할 때 사용자에게 추가 정보를 요청합니다. 호출 후 반드시 분석을 중단하세요.",
      {
        questions: z.array(
          z.object({
            question: z.string().describe("사용자에게 물어볼 질문"),
            options: z.array(
              z.object({
                label: z.string().describe("선택지 라벨"),
                description: z.string().describe("선택지 설명"),
              })
            ),
            multiSelect: z.boolean().describe("복수 선택 허용 여부"),
          })
        ),
      },
      async (args) => {
        pendingClarification = args.questions as ClarificationQuestion[];
        return {
          content: [
            {
              type: "text" as const,
              text: "사용자에게 추가 정보를 요청했습니다. 응답이 올 때까지 분석을 중단하세요.",
            },
          ],
        };
      }
    );

    const customServer = createSdkMcpServer({
      name: "musing-core",
      tools: [setQueryTypeTool, requestClarificationTool],
    });

    // ── PostToolUse Hook (실시간 trace 이벤트 emit) ───────
    const trackToolUse: HookCallback = async (hookInput) => {
      const inp = hookInput as {
        tool_name?: string;
        tool_input?: Record<string, unknown>;
      };
      const toolName = inp.tool_name ?? "";
      const toolInput = inp.tool_input ?? {};
      const elapsed = Date.now() - startMs;
      toolCallCount++;

      // "턴 N/MAX · 도구 K회" 접미사 — 모든 trace에 공통으로 붙입니다
      const turnSuffix = `턴 ${turnCount}/${env.claudeMaxTurns} · 도구 ${toolCallCount}회`;

      if (toolName.includes("readFileRange") || toolName.includes("read_file_range")) {
        filesRead++;
        const path = String(toolInput.path ?? "");
        const lineStart = Number(toolInput.lineStart ?? toolInput.line_start ?? 0);
        const locator = lineStart > 0 ? `${path}:${lineStart}` : path;
        if (locator) references.push({ locator, description: `파일 읽기: ${path}` });
        push({ event: "trace", data: { label: `저장소 탐색 (${turnSuffix})`, detail: path || undefined, elapsedMs: elapsed } });
      } else if (
        toolName.includes("searchCode") ||
        toolName.includes("search_code") ||
        toolName.includes("listFiles") ||
        toolName.includes("list_files")
      ) {
        const pattern = String(toolInput.pattern ?? toolInput.query ?? "");
        push({ event: "trace", data: { label: `저장소 탐색 (${turnSuffix})`, detail: pattern || undefined, elapsedMs: elapsed } });
      }

      if (toolName.includes("executeQuery") || toolName.includes("execute_query")) {
        dbQueries++;
        const table = String(toolInput.table ?? "");
        if (table) references.push({ locator: table, description: `DB 조회: ${table}` });
        push({ event: "trace", data: { label: `DB 조회 (${turnSuffix})`, detail: table || undefined, elapsedMs: elapsed } });
      } else if (
        toolName.includes("describeTable") ||
        toolName.includes("describe_table") ||
        toolName.includes("listTables") ||
        toolName.includes("list_tables")
      ) {
        push({ event: "trace", data: { label: `DB 조회 (${turnSuffix})`, elapsedMs: elapsed } });
      }

      return {};
    };

    // ── 프롬프트 구성 ─────────────────────────────────────
    // agentSessionId가 있으면 resume (되묻기 응답 or follow-up 모두 해당)
    const isResume = !!(session?.agentSessionId);
    let prompt: string;

    if (req.answers && Object.keys(req.answers).length > 0) {
      // 되묻기에 대한 답변
      const answerLines = Object.entries(req.answers)
        .map(([q, a]) => `  - "${q}" → ${a}`)
        .join("\n");
      prompt = `이전에 요청한 추가 정보에 대한 사용자 응답:\n${answerLines}\n\n원래 질문을 계속 분석해주세요.`;
    } else {
      // 신규 질문 또는 follow-up
      prompt = req.question;
    }

    // ── MCP 서버 설정 ─────────────────────────────────────
    const erpGitCmd = buildMcpCommand("src/mcp-servers/erp-git/index.ts");
    const erpDbCmd = buildMcpCommand("src/mcp-servers/erp-db/index.ts");

    const mcpServers: Record<
      string,
      { command: string; args: string[]; env?: Record<string, string> }
    > = {};

    if (process.env.HANWHA_ERP_GIT_URL) {
      mcpServers["erp-git"] = {
        ...erpGitCmd,
        env: { GIT_CACHE_DIR: env.gitCacheDir },
      };
    }

    if (env.erpDb.host && env.erpDb.name && env.erpDb.user) {
      mcpServers["erp-db"] = {
        ...erpDbCmd,
        env: {
          ERP_DB_HOST: env.erpDb.host,
          ERP_DB_PORT: String(env.erpDb.port),
          ERP_DB_NAME: env.erpDb.name,
          ERP_DB_USER: env.erpDb.user,
          ERP_DB_PASSWORD: env.erpDb.password,
          ERP_DB_QUERY_TIMEOUT_MS: String(env.dbQueryTimeoutMs),
        },
      };
    }

    // ── 에이전트 백그라운드 실행 ──────────────────────────
    const runAgent = async () => {
      let agentGen: ReturnType<typeof query> | null = null;
      let agentSessionId: string | undefined;
      let totalTokens = 0;

      // ── 턴(Turn) 추적 상태 ───────────────────────────────
      // 아젠틱 루프는 여러 LLM 턴으로 구성됩니다:
      //   Turn 1: 계획 텍스트 + 도구 호출 선언
      //   Turn 2~N-1: 추가 도구 호출 (중간 턴)
      //   Turn N: 최종 답변 (last turn)
      //
      // message_start 이벤트가 새 턴 시작을 알립니다.
      // content_block_delta / assistant 이벤트는 현재 턴에 귀속됩니다.
      //
      // 중간 턴 텍스트는 trace 이벤트로 emit하고,
      // 마지막 턴 텍스트만 token 이벤트로 스트리밍 + done.answer에 담습니다.

      /** 현재 활성 턴의 message ID (message_start에서 획득) */
      let currentMsgId: string | null = null;
      /** 현재 턴에 누적된 텍스트 */
      let currentTurnText = "";
      /** content_block_delta로 이미 스트리밍한 message ID 집합 (assistant 블록 중복 방지) */
      const streamedMsgIds = new Set<string>();

      /**
       * 현재 턴을 중간 턴으로 확정합니다.
       * @param elapsedMs - 경과 시간
       * @param turnNum   - 완료되는 턴 번호 (trace 레이블에 표시)
       */
      const finalizeIntermediateTurn = (elapsedMs: number, turnNum: number) => {
        if (currentMsgId && currentTurnText.trim()) {
          push({
            event: "trace",
            data: {
              label: `LLM 계획 (턴 ${turnNum}/${env.claudeMaxTurns})`,
              detail: currentTurnText.replace(/\n+/g, " ").substring(0, 120),
              elapsedMs,
            },
          });
        }
        currentMsgId = null;
        currentTurnText = "";
      };

      try {
        const queryOptions: Parameters<typeof query>[0] = {
          prompt,
          options: {
            systemPrompt: buildSystemPrompt(req.memberId),
            model: env.claudeModel,
            mcpServers: {
              ...mcpServers,
              "musing-core": customServer,
            },
            allowedTools: [
              "mcp__musing-core__set_query_type",
              "mcp__musing-core__request_clarification",
              "mcp__erp-git__listFiles",
              "mcp__erp-git__searchCode",
              "mcp__erp-git__readFileRange",
              "mcp__erp-db__listTables",
              "mcp__erp-db__describeTable",
              "mcp__erp-db__executeQuery",
            ],
            maxTurns: env.claudeMaxTurns,
            includePartialMessages: true,
            hooks: {
              PostToolUse: [{ matcher: ".*", hooks: [trackToolUse] }],
            },
            ...(isResume && session?.agentSessionId
              ? { resume: session.agentSessionId }
              : {}),
          },
        };

        agentGen = query(queryOptions);
        // 취소 지원: agentGen을 sessionStore에 등록
        this.sessionStore.registerRequest(req.requestId, agentGen as AsyncGenerator<unknown>);

        let maxTurnsHit = false; // error_max_turns 감지 플래그

        for await (const message of agentGen) {
          if (message && typeof message === "object") {
            const msg = message as Record<string, unknown>;

            // ── [DUMP] SDK 메시지 덤프 (턴 경계 확인용) ──────
            // msg.type과 msg.event?.type을 함께 출력합니다.
            // stream_event 하위 이벤트 타입까지 확인할 수 있습니다.
            const evtObj = msg.event as Record<string, unknown> | undefined;
            console.log(
              `[sdk-msg] type=${String(msg.type ?? "-")}` +
              ` event.type=${String(evtObj?.type ?? "-")}` +
              ` t=${Date.now() - startMs}ms`
            );

            // ── Agent SDK 세션 ID 캡처 (resume에 사용) ────────
            if (msg.type === "system" && msg.subtype === "init" && msg.session_id) {
              agentSessionId = String(msg.session_id);
            }

            // ── stream_event: 실시간 스트리밍 ────────────────
            // includePartialMessages: true 설정 시 SDK가 BetaRawMessageStreamEvent를
            // SDKPartialAssistantMessage.event에 담아 stream_event 타입으로 emit합니다.
            //
            // 이벤트 경로:
            //   msg.type === "stream_event"
            //     → msg.event.type === "message_start"   : 새 턴 시작, message ID 획득
            //     → msg.event.type === "content_block_delta"
            //         → msg.event.delta.type === "text_delta" : 토큰 emit
            if (msg.type === "stream_event" && evtObj) {

              // 새 턴 시작 (message_start)
              if (evtObj.type === "message_start") {
                const newMsgId = String(
                  (evtObj.message as Record<string, unknown> | undefined)?.id ?? ""
                );
                if (newMsgId && newMsgId !== currentMsgId) {
                  if (currentMsgId) {
                    // 직전 턴을 중간 턴으로 확정 → trace emit (현재 turnCount = 직전 턴 번호)
                    finalizeIntermediateTurn(Date.now() - startMs, turnCount);
                  }
                  turnCount++;   // 새 턴 번호 증가
                  currentMsgId = newMsgId;
                  currentTurnText = "";
                  // 이 message ID는 stream_event로 처리됨 → assistant 블록 중복 방지
                  streamedMsgIds.add(newMsgId);
                }
              }

              // 토큰 스트리밍 (content_block_delta → text_delta)
              if (evtObj.type === "content_block_delta") {
                const delta = evtObj.delta as Record<string, unknown> | undefined;
                if (delta?.type === "text_delta" && typeof delta.text === "string" && delta.text) {
                  currentTurnText += delta.text;
                  push({ event: "token", data: { text: delta.text } });
                }
              }
            }

            // ── assistant 완성 메시지 ─────────────────────────
            // stream_event로 이미 스트리밍된 msgId는 토큰을 다시 emit하지 않습니다.
            // includePartialMessages 없는 환경에서만 토큰을 emit하는 fallback 경로입니다.
            if (msg.type === "assistant") {
              const assistantMsg = msg.message as
                | { id?: string; content?: Array<{ type: string; text?: string }> }
                | undefined;
              const msgId = assistantMsg?.id ?? "";

              if (streamedMsgIds.has(msgId)) {
                // stream_event로 이미 처리 완료 → 중복 skip
              } else {
                // fallback: stream_event 없이 완성 메시지만 도착하는 환경
                if (msgId && currentMsgId && currentMsgId !== msgId) {
                  finalizeIntermediateTurn(Date.now() - startMs, turnCount);
                }
                if (msgId) currentMsgId = msgId;

                for (const block of assistantMsg?.content ?? []) {
                  if (block.type === "text" && typeof block.text === "string" && block.text) {
                    currentTurnText += block.text;
                    push({ event: "token", data: { text: block.text } });
                  }
                }
                if (msgId) streamedMsgIds.add(msgId);
              }
            }

            // ── result: 에이전트 루프 종료 ───────────────────
            // error_max_turns → 재시도 플래그만 세우고 루프 탈출 (에러 throw 안 함)
            // 그 외 비성공 subtype → 즉시 에러
            if (msg.type === "result") {
              const subtype = String(msg.subtype ?? "");
              if (subtype === "error_max_turns") {
                maxTurnsHit = true;
                break; // 루프 탈출 → 아래 재시도 블록으로 진행
              } else if (subtype !== "success") {
                const errDetail =
                  (msg.error as Record<string, unknown> | undefined)?.message ??
                  msg.result ??
                  `Agent 비정상 종료: ${subtype}`;
                throw new Error(String(errDetail));
              }
              // success: currentTurnText = 마지막 턴 텍스트 → done.answer
            }

            // ── 토큰 집계 ─────────────────────────────────────
            if (msg.usage) {
              const u = msg.usage as { input_tokens?: number; output_tokens?: number };
              totalTokens += (u.input_tokens ?? 0) + (u.output_tokens ?? 0);
            }
          }
        }

        // ── error_max_turns 재시도 ────────────────────────
        // 최대 턴 소진 시: 저장된 agentSessionId로 resume하여
        // 도구 없이 maxTurns:1로 한 번 더 호출 → 지금까지 확인한 내용만으로 답변 생성
        if (maxTurnsHit) {
          if (!agentSessionId) {
            throw new Error(
              `탐색 한도(${env.claudeMaxTurns}턴) 초과: agent 세션 ID 없어 재시도 불가`
            );
          }

          push({
            event: "trace",
            data: {
              label: `탐색 한도 도달 (${env.claudeMaxTurns}턴 소진)`,
              detail: "지금까지 확인한 내용으로 최종 답변 생성 중…",
              elapsedMs: Date.now() - startMs,
            },
          });

          // 재시도용 턴 상태 초기화 (토큰 누적 초기화)
          currentMsgId = null;
          currentTurnText = "";

          const retryGen = query({
            prompt:
              "지금까지 도구로 확인한 내용만을 토대로 추가 도구 호출 없이 최종 답변을 작성하세요.",
            options: {
              systemPrompt: buildSystemPrompt(req.memberId),
              model: env.claudeModel,
              mcpServers: { "musing-core": customServer },
              allowedTools: [], // 도구 호출 불허
              maxTurns: 1,
              includePartialMessages: true,
              resume: agentSessionId,
            },
          });

          try {
            for await (const retryMsg of retryGen) {
              if (retryMsg && typeof retryMsg === "object") {
                const rm = retryMsg as Record<string, unknown>;
                const revt = rm.event as Record<string, unknown> | undefined;

                // 재시도 토큰 스트리밍
                if (rm.type === "stream_event" && revt?.type === "content_block_delta") {
                  const delta = revt.delta as Record<string, unknown> | undefined;
                  if (delta?.type === "text_delta" && typeof delta.text === "string" && delta.text) {
                    currentTurnText += delta.text;
                    push({ event: "token", data: { text: delta.text } });
                  }
                }

                // assistant fallback (includePartialMessages 미지원 환경)
                if (rm.type === "assistant") {
                  const am = rm.message as
                    | { id?: string; content?: Array<{ type: string; text?: string }> }
                    | undefined;
                  const amid = am?.id ?? "";
                  if (!streamedMsgIds.has(amid)) {
                    for (const block of am?.content ?? []) {
                      if (block.type === "text" && typeof block.text === "string" && block.text) {
                        currentTurnText += block.text;
                        push({ event: "token", data: { text: block.text } });
                      }
                    }
                    if (amid) streamedMsgIds.add(amid);
                  }
                }

                // 재시도 result 체크 (성공이 아니면 에러)
                if (rm.type === "result" && String(rm.subtype ?? "") !== "success") {
                  throw new Error(`재시도 실패: ${String(rm.subtype ?? "unknown")}`);
                }

                if (rm.usage) {
                  const u = rm.usage as { input_tokens?: number; output_tokens?: number };
                  totalTokens += (u.input_tokens ?? 0) + (u.output_tokens ?? 0);
                }
              }
            }

            // 안내 문구를 답변 앞에 붙입니다
            const notice = "탐색 한도에 도달해 확인된 범위까지만 답변합니다.\n\n";
            currentTurnText = notice + currentTurnText.trim();

          } catch (retryErr) {
            throw new Error(
              `탐색 한도(${env.claudeMaxTurns}턴) 초과 후 최종 답변 생성 실패: ` +
              `${retryErr instanceof Error ? retryErr.message : String(retryErr)}`
            );
          }
        }

        // result 없이 generator가 끝난 경우에도 최선을 다해 답변을 구성합니다.
        // (일부 SDK 버전은 result를 emit하지 않을 수 있음)

        // 마지막 턴 텍스트 = done.answer
        const finalAnswer = currentTurnText.trim();

        // 세션 agentSessionId 저장 (다음 turn에 resume용)
        if (agentSessionId) {
          this.sessionStore.updateSession(sessionId, { agentSessionId });
        }

        // 최종 이벤트 emit
        if (pendingClarification) {
          this.sessionStore.updateSession(sessionId, {
            status: "waiting_input",
            queryType,
            pendingQuestions: pendingClarification,
          });
          push({
            event: "clarification",
            data: { questions: pendingClarification, type: queryType },
          });
        } else {
          this.sessionStore.updateSession(sessionId, { status: "done", queryType });
          push({
            event: "done",
            data: {
              answer: finalAnswer || "답변을 생성하지 못했습니다.",
              references,
              type: queryType,
              usage: {
                elapsedMs: Date.now() - startMs,
                tokens: totalTokens || undefined,
                filesRead,
                dbQueries,
              },
            },
          });
        }
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : String(err);
        this.sessionStore.updateSession(sessionId, { status: "failed" });
        push({ event: "error", data: { sessionId, error: errorMsg } });
      } finally {
        this.sessionStore.completeRequest(req.requestId);
        push(null); // 스트림 종료 신호
      }
    };

    // 백그라운드 실행 (await 없음 — 이벤트 큐로 통신)
    runAgent();

    // ── 이벤트 큐 → yield ─────────────────────────────────
    try {
      while (true) {
        if (queue.length === 0) {
          await new Promise<void>((r) => {
            waitResolve = r;
          });
        }
        const event = queue.shift()!;
        if (event === null) break;
        yield event;
      }
    } finally {
      cancelled = true; // 클라이언트 연결 끊김 시 push 차단
    }
  }
}
