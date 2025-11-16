"""
Redes Actor-Critic para PPO.

Actor: salida la distribución de acción (media para acciones continuas)
Critic: salida el valor V(s)

Diseñado para observaciones vectoriales (LIDAR) usando MLP.
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
    
    Usa MLP para observaciones vectoriales (no CNN).
    """
    
    def __init__(
        self,
        obs_shape,
        action_dim,
        action_type="discrete",
        hidden_size=256
    ):
        """
        Args:
            obs_shape: tuple, forma de la observación (n_features,) para vector
            action_dim: int, número de acciones (discreto) o dim del espacio (continuo)
            action_type: "discrete" o "continuous"
            hidden_size: tamaño de las capas ocultas de la MLP
        """
        super().__init__()
        
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.action_type = action_type
        
        # Calcular dimensión de entrada
        if len(obs_shape) == 1:
            # Vector 1D: (n_features,)
            input_dim = obs_shape[0]
        else:
            # Flatten si es multidimensional
            input_dim = int(np.prod(obs_shape))
        
        # Feature extractor: MLP para observaciones vectoriales
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Capa compartida (opcional, puede ser parte del feature extractor)
        # Por ahora lo dejamos como está para mantener estructura similar
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
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
        self._init_weights()  # Inicialización de pesos
    
    def _init_weights(self):
        """Inicializamos los pesos de la red."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Último layer del actor y critic con gain menor para estabilidad
        if self.action_type == "discrete":
            nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        else:
            nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
    
    def forward(self, obs):
        """
        Pasamos por la red MLP.
        Returns:
            features: features extraídas
        """
        # Asegurar que obs es un tensor en el mismo device que el modelo
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=next(self.parameters()).device)
        elif not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        
        # Flatten si es necesario
        if obs.ndim > 1:
            batch_size = obs.shape[0]
            obs = obs.view(batch_size, -1)
        else:
            obs = obs.view(1, -1)
        
        # Pasar por MLP
        features = self.feature_extractor(obs)
        features = self.shared_fc(features)
        return features
    
    def get_action_and_value(self, obs, action=None):
        """
        Evaluar observación(es) y opcionalmente una acción específica.
        
        Args:
            obs: tensor [batch, *obs_shape] o [*obs_shape]
            action: opcional, tensor [batch, action_dim] para evaluar
        
        Returns:
            action: acción seleccionada (si action=None)
            log_prob: log π(a|s)
            entropy: entropía de la distribución
            value: V(s)
        """
        # Asegurar que obs tiene batch dimension y está en el device correcto
        device = next(self.parameters()).device
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        elif isinstance(obs, torch.Tensor):
            obs = obs.to(device)
        
        if obs.ndim == len(self.obs_shape):
            obs = obs.unsqueeze(0)
        
        features = self.forward(obs)
        
        # Critic: predice el valor V(s)
        value = self.critic_head(features).squeeze(-1)  # [batch]
        
        # Actor: predice la política π(a|s)
        if self.action_type == "discrete":
            logits = self.actor_head(features)  # [batch, action_dim]
            dist = Categorical(logits=logits)
            
            if action is None:
                action = dist.sample()
            else:
                # Asegurar que action es un tensor con la forma correcta
                if isinstance(action, (int, np.integer)):
                    action = torch.tensor([action], device=obs.device)
                elif isinstance(action, np.ndarray):
                    action = torch.as_tensor(action, dtype=torch.long, device=obs.device)
                if action.ndim == 0:
                    action = action.unsqueeze(0)
            
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
        else:  # acciones continuas
            mean = self.actor_mean(features)  # [batch, action_dim]
            std = torch.exp(self.log_std)  # broadcast a [action_dim]
            dist = Normal(mean, std)
            
            if action is None:
                action = dist.sample()
            else:
                if isinstance(action, np.ndarray):
                    action = torch.as_tensor(action, dtype=torch.float32, device=obs.device)
                if action.ndim == 1 and len(action.shape) == 1:
                    action = action.unsqueeze(0)
            
            log_prob = dist.log_prob(action).sum(dim=-1)  # sumamos sobre las dimensiones de la acción
            entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value 
    
    def get_value(self, obs):
        """Solo obtenemos V(s) - más eficiente cuando no necesitamos la política."""
        # Asegurar que obs tiene batch dimension y está en el device correcto
        device = next(self.parameters()).device
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        elif isinstance(obs, torch.Tensor):
            obs = obs.to(device)
        
        if obs.ndim == len(self.obs_shape):
            obs = obs.unsqueeze(0)
        
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
