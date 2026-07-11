import { Routes } from '@angular/router';
import { HomeComponent } from './home/home';
import { ChatComponent } from './chat/chat';

export const routes: Routes = [
  { path: '', component: HomeComponent, title: 'UltimateAI | Home' },
  { path: 'chat', component: ChatComponent, title: 'UltimateAI | Chat' },
  { path: '**', redirectTo: '' }
];