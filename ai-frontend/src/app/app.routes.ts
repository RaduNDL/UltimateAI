import { Routes } from '@angular/router';
import { HomeComponent } from './home/home';
import { ChatComponent } from './chat/chat';

export const routes: Routes = [
  { path: '', component: HomeComponent, title: 'UltimateAI | Home' },
  { path: 'chat', component: ChatComponent, title: 'UltimateAI | Chat' },
  { 
    path: 'login', 
    loadComponent: () => import('./login/login').then(m => m.LoginComponent), 
    title: 'UltimateAI | Login' 
  },
  { 
    path: 'register', 
    loadComponent: () => import('./register/register').then(m => m.RegisterComponent), 
    title: 'UltimateAI | Register' 
  },
  { path: '**', redirectTo: '' }
];