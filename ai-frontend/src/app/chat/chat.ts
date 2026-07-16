import {
  ChangeDetectorRef,
  Component,
  ElementRef,
  Inject,
  OnDestroy,
  OnInit,
  PLATFORM_ID,
  ViewChild,
  ViewEncapsulation
} from '@angular/core';
import { CommonModule, isPlatformBrowser } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { timeout } from 'rxjs';
import { AuthService } from '../core/auth.service';

type Role = 'user' | 'assistant';
type ResponseMode = 'auto' | 'fast' | 'detailed';

interface SourceReference {
  path: string;
  start_line: number;
  end_line: number;
}

interface ChatMessage {
  role: Role;
  content: string;
  timestamp: string;
  introductory?: boolean;
  mode?: string;
  sources?: SourceReference[];
}

interface HistoryMessage {
  role: Role;
  content: string;
}

interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_id?: string;
  quantization?: string;
  adapter_loaded?: boolean;
  rag?: { chunks: number };
}

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.html',
  styleUrl: './chat.css',
  encapsulation: ViewEncapsulation.None
})
export class ChatComponent implements OnInit, OnDestroy {
  @ViewChild('scrollWrap')
  scrollWrap!: ElementRef<HTMLDivElement>;

  readonly apiBaseUrl = 'http://127.0.0.1:8000';
  prompt = '';
  loading = false;
  error = '';
  backendOnline = false;
  modelLabel = 'Backend offline';
  activeRoute = '';
  projects: string[] = [];
  selectedProject = '';
  useProjectContext = false;
  showChatOptions = false;
  private abortController: AbortController | null = null;
  private healthTimer: number | null = null;
  private readonly isBrowser: boolean;

  settings = {
    max_new_tokens: 384,
    temperature: 0.35,
    top_p: 0.9,
    repetition_penalty: 1.08,
    response_mode: 'auto' as ResponseMode
  };

  messages: ChatMessage[] = [this.createWelcomeMessage()];

  constructor(
    private readonly http: HttpClient,
    private readonly sanitizer: DomSanitizer,
    private readonly changeDetector: ChangeDetectorRef,
    private readonly auth: AuthService,
    @Inject(PLATFORM_ID) platformId: object
  ) {
    this.isBrowser = isPlatformBrowser(platformId);
  }

  ngOnInit(): void {
    if (!this.isBrowser) {
      return;
    }
    this.checkBackend();
    this.loadProjects();
    this.healthTimer = window.setInterval(() => this.checkBackend(), 10_000);
  }

  ngOnDestroy(): void {
    if (this.healthTimer !== null) {
      window.clearInterval(this.healthTimer);
    }
    this.abortGeneration();
  }

  private createWelcomeMessage(): ChatMessage {
    return {
      role: 'assistant',
      content: 'Hello! I am UltimateAI, your English-only local software engineering assistant. Select a project only when you need help with its code, then send the code, error, or feature.',
      timestamp: this.now(),
      introductory: true
    };
  }

  checkBackend(): void {
    if (!this.isBrowser) {
      return;
    }
    this.http.get<HealthResponse>(`${this.apiBaseUrl}/health`).pipe(timeout(5_000)).subscribe({
      next: health => {
        this.backendOnline = health.status === 'ok' && health.model_loaded;
        const model = health.model_id?.split('/').pop() || 'Local model';
        const adapter = health.adapter_loaded ? 'LoRA adapter' : 'frozen base';
        const rag = health.rag?.chunks ? `RAG ${health.rag.chunks} chunks` : 'RAG unavailable';
        this.modelLabel = `${model} | ${adapter} | ${health.quantization || 'local'} | ${rag}`;
        this.changeDetector.markForCheck();
      },
      error: () => {
        this.backendOnline = false;
        this.modelLabel = 'Backend offline - start FastAPI';
        this.changeDetector.markForCheck();
      }
    });
  }

  loadProjects(): void {
    if (!this.isBrowser) {
      return;
    }
    this.http.get<{ projects: string[] }>(`${this.apiBaseUrl}/projects`).pipe(timeout(5_000)).subscribe({
      next: result => {
        this.projects = result.projects || [];
        this.changeDetector.markForCheck();
      },
      error: () => {
        this.projects = [];
        this.changeDetector.markForCheck();
      }
    });
  }

  now(): string {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  trackByIndex(index: number): number {
    return index;
  }

  renderMessage(content: string): SafeHtml {
    const parts = content.split(/(```[\s\S]*?```)/g);
    let html = '';
    for (const part of parts) {
      if (part.startsWith('```')) {
        const code = part.replace(/^```[\w#+.-]*\n?/, '').replace(/\n?```$/, '');
        html += `<pre class="code-block"><code>${this.escapeHtml(code)}</code></pre>`;
      } else {
        const escaped = this.escapeHtml(part)
          .replace(/`([^`\n]+)`/g, '<code class="inline-code">$1</code>')
          .replace(/\n/g, '<br>');
        html += `<span>${escaped}</span>`;
      }
    }
    return this.sanitizer.bypassSecurityTrustHtml(html);
  }

  private escapeHtml(value: string): string {
    return value
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  onInput(event: Event): void {
    const textarea = event.target as HTMLTextAreaElement;
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
  }

  clearChat(): void {
    if (this.loading) {
      return;
    }
    this.messages = [this.createWelcomeMessage()];
    this.prompt = '';
    this.error = '';
    this.activeRoute = '';
    this.showChatOptions = false;
    this.scrollToBottom();
  }

  toggleChatOptions(): void {
    if (!this.loading) {
      this.showChatOptions = !this.showChatOptions;
    }
  }

  send(): void {
    void this.sendStream();
  }

  private async sendStream(): Promise<void> {
    const text = this.prompt.trim();
    if (!text || this.loading) {
      return;
    }

    const history: HistoryMessage[] = this.messages
      .filter(message => !message.introductory && message.content.trim())
      .map(message => ({ role: message.role, content: message.content }))
      .slice(-8);

    this.messages.push({ role: 'user', content: text, timestamp: this.now() });
    const assistantIndex = this.messages.length;
    this.messages.push({ role: 'assistant', content: '', timestamp: this.now() });
    this.prompt = '';
    this.error = '';
    this.loading = true;
    this.activeRoute = '';
    this.abortController = new AbortController();
    this.scrollToBottom();

    try {
      const response = await fetch(`${this.apiBaseUrl}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...this.auth.authorizationHeaders() },
        signal: this.abortController.signal,
        body: JSON.stringify({
          prompt: text,
          history,
          max_new_tokens: this.settings.max_new_tokens,
          temperature: this.settings.temperature,
          top_p: this.settings.top_p,
          repetition_penalty: this.settings.repetition_penalty,
          response_mode: this.settings.response_mode,
          project: this.selectedProject || null,
          use_project_context: this.useProjectContext
        })
      });

      if (!response.ok || !response.body) {
        const detail = await response.text();
        throw new Error(detail || `Backend error ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let finished = false;

      while (!finished) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';
        for (const event of events) {
          const line = event.split('\n').find(value => value.startsWith('data: '));
          if (!line) {
            continue;
          }
          const payload = JSON.parse(line.slice(6)) as {
            type: string;
            text?: string;
            detail?: string;
            mode?: string;
            sources?: SourceReference[];
          };
          if (payload.type === 'token') {
            this.messages[assistantIndex].content += payload.text || '';
            this.scrollToBottom();
          } else if (payload.type === 'done') {
            this.messages[assistantIndex].mode = payload.mode;
            this.messages[assistantIndex].sources = payload.sources || [];
            this.activeRoute = payload.mode || '';
            finished = true;
          } else if (payload.type === 'error') {
            throw new Error(payload.detail || 'Generation failed.');
          }
          this.changeDetector.markForCheck();
        }
      }

      if (!this.messages[assistantIndex].content.trim()) {
        this.messages[assistantIndex].content = 'I did not receive a useful response. Rephrase the question or include the relevant code.';
      }
      this.backendOnline = true;
    } catch (error) {
      const aborted = error instanceof DOMException && error.name === 'AbortError';
      this.messages[assistantIndex].content = aborted
        ? 'Generation stopped.'
        : 'I could not contact the backend. Check that FastAPI is running and try again.';
      this.error = aborted ? '' : (error instanceof Error ? error.message : 'Backend unavailable.');
      if (!aborted) {
        this.backendOnline = false;
      }
    } finally {
      this.loading = false;
      this.abortController = null;
      this.changeDetector.markForCheck();
      this.scrollToBottom();
    }
  }

  abortGeneration(): void {
    this.abortController?.abort();
  }

  onEnter(event: Event): void {
    const keyboardEvent = event as KeyboardEvent;
    if (keyboardEvent.shiftKey) {
      return;
    }
    keyboardEvent.preventDefault();
    this.send();
  }

  private scrollToBottom(): void {
    setTimeout(() => {
      if (this.scrollWrap) {
        this.scrollWrap.nativeElement.scrollTop = this.scrollWrap.nativeElement.scrollHeight;
      }
    }, 0);
  }
}
