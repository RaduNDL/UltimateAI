import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, tap } from 'rxjs';

export interface AuthUser {
  id: string;
  email: string;
  username: string;
  display_name: string;
  roles: string[];
}

interface AuthResponse {
  access_token: string;
  token_type: 'bearer';
  expires_at: string;
  user: AuthUser;
}

export interface RegisterPayload {
  display_name: string;
  username: string;
  email: string;
  password: string;
}

export interface LoginPayload {
  email: string;
  password: string;
}

@Injectable({ providedIn: 'root' })
export class AuthService {
  private readonly apiBaseUrl = 'http://127.0.0.1:8000';
  private readonly tokenKey = 'ultimateai.access_token';
  private readonly userKey = 'ultimateai.user';
  private readonly userSubject = new BehaviorSubject<AuthUser | null>(this.readStoredUser());

  readonly user$ = this.userSubject.asObservable();

  constructor(private readonly http: HttpClient) {}

  get user(): AuthUser | null {
    return this.userSubject.value;
  }

  get accessToken(): string | null {
    return this.storage()?.getItem(this.tokenKey) ?? null;
  }

  authorizationHeaders(): Record<string, string> {
    const token = this.accessToken;
    return token ? { Authorization: `Bearer ${token}` } : {};
  }

  register(payload: RegisterPayload): Observable<AuthResponse> {
    return this.http.post<AuthResponse>(`${this.apiBaseUrl}/auth/register`, payload).pipe(
      tap(response => this.storeSession(response))
    );
  }

  login(payload: LoginPayload): Observable<AuthResponse> {
    return this.http.post<AuthResponse>(`${this.apiBaseUrl}/auth/login`, payload).pipe(
      tap(response => this.storeSession(response))
    );
  }

  logout(): void {
    const storage = this.storage();
    storage?.removeItem(this.tokenKey);
    storage?.removeItem(this.userKey);
    this.userSubject.next(null);
  }

  private storeSession(response: AuthResponse): void {
    const storage = this.storage();
    storage?.setItem(this.tokenKey, response.access_token);
    storage?.setItem(this.userKey, JSON.stringify(response.user));
    this.userSubject.next(response.user);
  }

  private readStoredUser(): AuthUser | null {
    try {
      const raw = this.storage()?.getItem(this.userKey);
      return raw ? JSON.parse(raw) as AuthUser : null;
    } catch {
      return null;
    }
  }

  private storage(): Storage | null {
    return typeof localStorage === 'undefined' ? null : localStorage;
  }
}
