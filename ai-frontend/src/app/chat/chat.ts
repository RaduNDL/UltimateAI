import { Component, ElementRef, ViewChild, AfterViewChecked } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

interface ChatMessage {
  text: string;
  role: 'user' | 'ai';
}

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.html',
  styleUrls: ['./chat.css'],
})
export class ChatComponent implements AfterViewChecked {
  @ViewChild('chatMessages') private chatScrollContainer!: ElementRef<HTMLDivElement>;

  messages: ChatMessage[] = [];
  userInput = '';
  isThinking = false;

  ngAfterViewChecked(): void {
    this.scrollToBottom();
  }

  private scrollToBottom(): void {
    if (!this.chatScrollContainer?.nativeElement) return;
    const el = this.chatScrollContainer.nativeElement;
    el.scrollTop = el.scrollHeight;
  }

  async sendMessage(): Promise<void> {
    const text = this.userInput.trim();
    if (!text || this.isThinking) return;

    this.messages.push({ text, role: 'user' });
    this.userInput = '';
    this.isThinking = true;

    const thinkingIndex = this.messages.push({ text: 'Se gândește...', role: 'ai' }) - 1;

    try {
      const response = await fetch('http://127.0.0.1:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: text, max_new_tokens: 180, temperature: 0.8, top_k: 50 }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data: { response?: string } = await response.json();
      this.messages[thinkingIndex].text = (data.response?.trim() || 'Modelul nu a returnat text.');
    } catch {
      this.messages[thinkingIndex].text =
        'Eroare conexiune la backend. Verifică dacă FastAPI rulează pe http://127.0.0.1:8000';
    } finally {
      this.isThinking = false;
    }
  }

  handleKeyPress(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void this.sendMessage();
    }
  }
}