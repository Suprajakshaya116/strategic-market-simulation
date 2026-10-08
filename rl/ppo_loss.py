import torch
import torch.nn.functional as F


def compute_ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    clip_eps: float = 0.2,
    c1_value_loss_coeff: float = 0.5,
    c2_entropy_coeff: float = 0.01,
    clip_value_loss: bool = True
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute complete PPO loss with clipped surrogate, value loss, entropy bonus, and diagnostic metrics.

    PPO Loss Equations:
      1. Ratio: r_t(theta) = exp(new_log_prob - old_log_prob)
      2. Clipped Surrogate Policy Loss:
         L_CLIP = E[ min( r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t ) ]
      3. Value Function Loss (optional value clipping):
         L_VF = 0.5 * E[ (V_theta(s_t) - V_target)^2 ]
      4. Entropy Bonus:
         L_ENT = E[ Entropy(pi_theta) ]
      5. Total Loss:
         L_TOTAL = - L_CLIP + c1 * L_VF - c2 * L_ENT
    """
    # 1. Policy Loss
    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # 2. Value Function Loss
    if clip_value_loss:
        v_clipped = old_values + torch.clamp(
            new_values - old_values,
            -clip_eps,
            clip_eps
        )
        vf_loss1 = (new_values - returns) ** 2
        vf_loss2 = (v_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
    else:
        value_loss = 0.5 * F.mse_loss(new_values, returns)

    # 3. Entropy Loss (Negative entropy bonus)
    entropy_loss = -entropy.mean()

    # Total combined loss
    total_loss = policy_loss + c1_value_loss_coeff * value_loss + c2_entropy_coeff * entropy_loss

    # Diagnostics / Metrics
    with torch.no_grad():
        approx_kl = ((ratio - 1) - log_ratio).mean().item()
        clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()

    metrics = {
        "total_loss": total_loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.mean().item(),
        "approx_kl": approx_kl,
        "clip_frac": clip_frac
    }

    return total_loss, metrics
