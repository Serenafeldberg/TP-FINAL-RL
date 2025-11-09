"""
Redes Actor-Critic para PPO.

Actor: salida la distribución de acción (media para acciones continuas)
Critic: salida el valor V(s)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np


class ActorCritic(nn.Module):
    """
    - Actor: predice la política π(a|s)
    - Critic: predice el valor V(s), para criticar la acción del actor
    """
    
    def __init__(
        self,
        obs_shape,
        action_dim,
        action_type="discrete",
        cnn_channels=(32, 64, 64),
        cnn_kernels=(8, 4, 3),
        cnn_strides=(4, 2, 1),
        hidden_size=512
    ):
        """
        Args:
            obs_shape: tuple, forma de la observación (C, H, W) para imágenes
            action_dim: int, número de acciones (discreto) o dim del espacio (continuo)
            action_type: "discrete" o "continuous"
            cnn_channels: canales de las capas conv
            cnn_kernels: tamaños de kernel
            cnn_strides: strides
            hidden_size: tamaño de la capa oculta final
        """
        super().__init__()
        
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.action_type = action_type
        
        # Feature extractor (CNN para imágenes tipo Atari)
        if len(obs_shape) == 3:  # (C, H, W)
            self.feature_extractor = self._build_cnn(
                obs_shape[0], cnn_channels, cnn_kernels, cnn_strides
            )
            # Calcular tamaño de salida de CNN
            with torch.no_grad():
                dummy = torch.zeros(1, *obs_shape)
                cnn_out = self.feature_extractor(dummy)
                cnn_out_size = cnn_out.view(1, -1).size(1)
        else:
            # Para estados de baja dimensión
            self.feature_extractor = nn.Identity()
            cnn_out_size = obs_shape[0] if len(obs_shape) == 1 else np.prod(obs_shape)
        
        # Capa compartida
        self.shared_fc = nn.Sequential(
            nn.Linear(cnn_out_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head
        if action_type == "discrete":
            self.actor_head = nn.Linear(hidden_size, action_dim)
        else:
            # Para acciones continuas: predecimos la media de una Normal
            self.actor_mean = nn.Linear(hidden_size, action_dim)
            # Log std como parámetro aprendible
            # Inicializamos en 0 → std = 1
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head
        self.critic_head = nn.Linear(hidden_size, 1)
        self._init_weights() # Inicialización de pesos
    
    def _build_cnn(self, in_channels, channels, kernels, strides):
        """Construimos el extractor de características CNN."""
        layers = []
        prev_c = in_channels
        
        for c, k, s in zip(channels, kernels, strides):
            layers.append(nn.Conv2d(prev_c, c, kernel_size=k, stride=s))
            layers.append(nn.ReLU())
            prev_c = c
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Inicializamos los pesos de la red."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Último layer del actor y critic con gain menor para estabilidad
        # Gain de 0.01 es demasiado bajo y hace la política casi determinista
        if self.action_type == "discrete":
            nn.init.orthogonal_(self.actor_head.weight, gain=0.1)  # Aumentado de 0.01
        else:
            nn.init.orthogonal_(self.actor_mean.weight, gain=0.1)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
    
    def forward(self, obs):
        """
        Pasamos por la red.
        Returns:
            features: features extraídas
            x: salida de la red
        """
        x = self.feature_extractor(obs)
        x = x.view(x.size(0), -1)  # flatten para que sea un vector
        features = self.shared_fc(x)
        return features
    
    def get_action_and_value(self, obs, action=None):
        """
        Evaluar observación(es) y opcionalmente una acción específica.
        
        Args:
            obs: tensor [batch, *obs_shape]
            action: opcional, tensor [batch, action_dim] para evaluar
        
        Returns:
            action: acción seleccionada (si action=None)
            log_prob: log π(a|s)
            entropy: entropía de la distribución
            value: V(s)
        """
        features = self.forward(obs)
        
        # Critic: predice el valor V(s)
        value = self.critic_head(features).squeeze(-1)  # [batch]
        
        # Actor: predice la política π(a|s)
        if self.action_type == "discrete":
            logits = self.actor_head(features)  # [batch, action_dim]
            dist = Categorical(logits=logits) #modelo las acciones con una distribucion categorica, podria ser una normal
            
            if action is None:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
        else:  # acciones continuas
            mean = self.actor_mean(features)  # [batch, action_dim]
            std = torch.exp(self.log_std)  # broadcast a [action_dim]
            dist = Normal(mean, std)
            
            if action is None:
                action = dist.sample()
            
            log_prob = dist.log_prob(action).sum(dim=-1)  # sumamos sobre las dimensiones de la acción
            entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value 
    
    def get_value(self, obs):
        """Solo obtenemos V(s) - más eficiente cuando no necesitamos la política."""
        features = self.forward(obs)
        return self.critic_head(features).squeeze(-1)


class FeedForwardNN(nn.Module):
    """
    Red MLP simple (para estados de baja dimensión)
    """
    
    def __init__(self, in_dim, out_dim, hidden_size=64):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, out_dim)
    
    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        
        x = F.relu(self.layer1(obs))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
