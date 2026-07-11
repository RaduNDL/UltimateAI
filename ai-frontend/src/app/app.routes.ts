import { Routes } from '@angular/router';
import { HomeComponent } from './home/home';
import { ChatComponent } from './chat/chat';

export const routes: Routes = [
  { path: '', component: HomeComponent },
  { path: 'chat', component: ChatComponent },
  { path: '**', redirectTo: '' }
];