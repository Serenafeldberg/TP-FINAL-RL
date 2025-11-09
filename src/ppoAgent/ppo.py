import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple
import time


class PPO:
    """
    Algoritmo PPO-Clip.
    Componentes principales:
    1. Rollout: recolectamos trayectorias con la política actual
    2. Computamos las advantages usando GAE
    3. Actualizamos la política maximizando la surrogate objective clipped
    4. Actualizamos la función de valor minimizando el MSE (Mean Squared Error)
    """
    
    def __init__(
        self,
        actor_critic,
        learning_rate=2.5e-4,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        lr_decay=True,
        device="cpu"
    ):
        """
        Args:
            actor_critic: red Actor-Critic
            learning_rate: learning rate inicial
            clip_epsilon: ε para el clipping de PPO
            value_loss_coef: coeficiente para la loss del critic
            entropy_coef: coeficiente para el bonus de entropía
            max_grad_norm: norm máximo para gradient clipping
            lr_decay: si usar linear annealing del LR
            device: 'cpu' o 'cuda'
        """
        self.actor_critic = actor_critic.to(device)
        self.device = device
        
        # Hiperparámetros
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.lr_decay = lr_decay
        self.initial_lr = learning_rate
        
        # Optimizer (uso Adam como en el paper)
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=learning_rate,
            eps=1e-5  # valor del paper original de PPO de 2017
        )
        
        # Para logging
        self.n_updates = 0
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False):
        """
        Seleccionamos una acción dada una observación.
        
        Args:
            obs: observación (single)
            deterministic: si True, tomamos la media (para evaluación)
        
        Returns:
            action: acción 
            log_prob: log π(a|s)
            value: V(s)
        """
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            if obs_tensor.ndim == len(self.actor_critic.obs_shape):
                obs_tensor = obs_tensor.unsqueeze(0)  
            
            if deterministic:
                # evaluación: tomamos la moda/media
                if self.actor_critic.action_type == "discrete":
                    features = self.actor_critic.forward(obs_tensor)
                    logits = self.actor_critic.actor_head(features)
                    action = torch.argmax(logits, dim=-1)
                else:
                    features = self.actor_critic.forward(obs_tensor)
                    action = self.actor_critic.actor_mean(features)
                
                log_prob = torch.zeros(1, device=self.device)
                value = self.actor_critic.critic_head(features).squeeze(-1)
            else:
                # sampleamos de la política vieja
                action, log_prob, entropy, value = self.actor_critic.get_action_and_value(obs_tensor)
            
            if self.actor_critic.action_type == "discrete":
                action = action.item()
            else:
                action = action.squeeze(0).cpu().numpy()
            
            log_prob = log_prob.item() if log_prob.numel() == 1 else log_prob.cpu().numpy()
            value = value.item()
        
        return action, log_prob, value
    
    def evaluate_actions(self, obs, actions):
        """
        Evaluamos las acciones bajo la política actual.
        Returns:
            log_probs: log π(a|s) [B]
            entropy: entropía [B]
            values: V(s) [B]
        """
        _, log_probs, entropy, values = self.actor_critic.get_action_and_value(obs, actions)
        return log_probs, entropy, values
    
    def update(
        self,
        rollout_buffer,
        batch_size: int,
        n_epochs: int,
        total_timesteps: Optional[int] = None,
        current_timestep: Optional[int] = None
    ):
        """
        Actualizar la política usando los datos del rollout buffer.
        
        Args:
            rollout_buffer: buffer con las trayectorias
            batch_size: tamaño de mini-batch
            n_epochs: número de épocas de entrenamiento sobre los datos
            total_timesteps: timesteps totales (para el annealing del learning rate)
            current_timestep: timestep actual (para el annealing del learning rate)
        
        Returns:
            dict con métricas de entrenamiento
        """
        # annealing del learning rate
        if self.lr_decay and total_timesteps is not None and current_timestep is not None:
            frac = 1.0 - (current_timestep / total_timesteps)
            new_lr = self.initial_lr * frac
            new_lr = max(new_lr, 0.0)  # no negativo para evitar problemas de estabilidad
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        
        policy_losses = []
        value_losses = []
        entropy_losses = []
        clip_fractions = []
        approx_kls = []
        
        # entrenamos por n_epochs
        for epoch in range(n_epochs):
            # iteramos sobre los mini-batches
            for batch in rollout_buffer.get(batch_size):
                obs = batch["observations"]
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["values"]
                
                # evaluamos las acciones con la política actual
                log_probs, entropy, values = self.evaluate_actions(obs, actions)
                
                # Policy Loss (surrogate clipped)
                # ratio = π_new / π_old = exp(log π_new - log π_old)
                ratios = torch.exp(log_probs - old_log_probs)
                
                # Surrogate losses
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                
                # Policy loss: -E[min(surr1, surr2)]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss 
                # clipped value loss ( ayuda a estabilidad)
                values_clipped = old_values + torch.clamp(
                    values - old_values,
                    -self.clip_epsilon,
                    self.clip_epsilon
                )
                value_loss_unclipped = (values - returns).pow(2)
                value_loss_clipped = (values_clipped - returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                
                # Entropy Bonus 
                entropy_loss = -entropy.mean()
                
                # Total Loss 
                # L = L_policy + c1*L_value - c2*H(π)
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # gradient clipping 
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # logging
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                with torch.no_grad():
                    clip_frac = torch.mean((torch.abs(ratios - 1.0) > self.clip_epsilon).float()).item()
                    clip_fractions.append(clip_frac)
                    
                    # aproximación de la divergencia KL
                    approx_kl = ((ratios - 1) - log_probs + old_log_probs).mean().item()
                    approx_kls.append(approx_kl)
        
        self.n_updates += 1
        
        # retornamos las métricas
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "clip_fraction": np.mean(clip_fractions),
            "approx_kl": np.mean(approx_kls),
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }
    
    def save(self, path: str):
        """guardamos el modelo."""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'n_updates': self.n_updates,
        }, path)
        print(f"Modelo guardado en: {path}")
    
    def load(self, path: str):
        """cargamos el modelo."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.n_updates = checkpoint.get('n_updates', 0)
        print(f"Modelo cargado desde: {path}")
