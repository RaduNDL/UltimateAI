import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { HttpErrorResponse } from '@angular/common/http';
import { Router, RouterLink } from '@angular/router';
import { AuthService } from '../core/auth.service';

@Component({
  selector: 'app-register',
  standalone: true,
  imports: [FormsModule, CommonModule, RouterLink],
  templateUrl: './register.html',
  styleUrl: './register.css'
})
export class RegisterComponent {
  name = '';
  username = '';
  email = '';
  password = '';
  errorMessage = '';
  loading = false;

  constructor(
    private readonly auth: AuthService,
    private readonly router: Router
  ) {}

  onSubmit(): void {
    this.errorMessage = '';
    if (!this.name || !this.username || !this.email || !this.password) {
      this.errorMessage = 'Please fill in all fields.';
      return;
    }
    if (!/^[A-Za-z0-9_]{3,40}$/.test(this.username)) {
      this.errorMessage = 'Username must contain 3-40 letters, numbers, or underscores.';
      return;
    }
    if (this.password.length < 8) {
      this.errorMessage = 'Password must contain at least 8 characters.';
      return;
    }

    this.loading = true;
    this.auth.register({
      display_name: this.name,
      username: this.username,
      email: this.email,
      password: this.password
    }).subscribe({
      next: () => void this.router.navigateByUrl('/'),
      error: (error: HttpErrorResponse) => {
        this.errorMessage = error.error?.detail || 'Unable to create the account. Try again.';
        this.loading = false;
      }
    });
  }
}
