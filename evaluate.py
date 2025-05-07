
import argparse
import torch
import inspect
import re
import numpy as np

from dqn import QNetwork
from env1 import Env1

def inspect_pth(path: str, map_location: str = "cpu"):
    """
    Load and display contents of a .pth checkpoint.
    """
    data = torch.load(path, map_location=map_location)
    print(f"[inspect] Loaded object of type: {type(data)}")
    if isinstance(data, dict):
        print("[inspect] Top-level keys:")
        for k in data:
            print(f"  • {k}")
        if all(isinstance(v, torch.Tensor) for v in data.values()):
            print("\n[inspect] Detected state_dict → layer shapes:")
            for name, tensor in data.items():
                print(f"  • {name:40s} → {tuple(tensor.shape)}")
    else:
        print("[inspect] Raw contents:", data)
    return data


def infer_dims_from_state_dict(sd: dict):
    """
    Infer (state_dim, action_dim) by scanning net.<i>.weight shapes.
    """
    pat = re.compile(r"^net\.(\d+)\.weight$")
    layers = {}
    for k, v in sd.items():
        m = pat.match(k)
        if m:
            layers[int(m.group(1))] = tuple(v.shape) 
    if not layers:
        raise ValueError("No net.<i>.weight keys found in state_dict")
    first, last = min(layers), max(layers)
    out_first, in_first = layers[first]
    out_last, in_last = layers[last]
    return in_first, out_last


def dummy_evaluate(model: torch.nn.Module, input_dim: int, runs: int = 3):
    """
    Perform a few random forward passes to sanity-check the model.
    """
    print(f"\n[dummy eval] {runs} runs (input_dim={input_dim})")
    for i in range(1, runs+1):
        x = torch.randn(1, input_dim)
        y = model(x)
        print(f"  • run {i:>2}: output shape={tuple(y.shape)}")


def evaluate_on_env(env, model: torch.nn.Module, state_dim: int, device: torch.device, num_eps: int = 10):
    """
    Greedy evaluation on a custom Env with action masking.
    """
    returns = []
    print(f"\n[evaluate] Running {num_eps} episodes on custom Env")

    for ep in range(1, num_eps + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            obs_arr = np.array(obs, dtype=np.float32).flatten()
            if obs_arr.size != state_dim:
                raise RuntimeError(
                    f"Obs size {obs_arr.size} != expected {state_dim}."
                )
            state = torch.from_numpy(obs_arr).unsqueeze(0).to(device)


            with torch.no_grad():
                q_vals = model(state) 


            valid = env.getValidActions(env.current_player).astype(bool)
            mask = torch.from_numpy(valid).to(device).unsqueeze(0)
            masked_q = q_vals.masked_fill(~mask, -float('inf'))
            action = int(masked_q.argmax(dim=1).item())


            obs, reward, done, _ = env.step(action)
            total_reward += reward

        returns.append(total_reward)
        print(f"  • Episode {ep:2d}: Return = {total_reward}")

    avg = sum(returns) / len(returns)
    print(f"Average return: {avg}")
    return returns


def main():
    parser = argparse.ArgumentParser(
        description="Load a .pth and evaluate on custom Env"
    )
    parser.add_argument(
        "--model-path", type=str, default="dqn_model.pth",
        help="Path to the .pth checkpoint"
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--dummy-runs", type=int, default=3,
        help="Number of sanity-forward passes"
    )
    args = parser.parse_args()


    ckpt = inspect_pth(args.model_path)
    sd = ckpt.get("state_dict", ckpt)
    state_dim, action_dim = infer_dims_from_state_dict(sd)
    print(f"\n[infer] state_dim={state_dim}, action_dim={action_dim}\n")


    from dqn import QNetwork
    model = QNetwork(state_dim, action_dim)
    model.load_state_dict(sd)
    model.eval()
    device = torch.device("cpu")


    dummy_evaluate(model, input_dim=state_dim, runs=args.dummy_runs)


    env = Env1()
    evaluate_on_env(env, model, state_dim, device, num_eps=args.episodes)

if __name__ == "__main__":
    main()
