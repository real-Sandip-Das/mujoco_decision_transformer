import argparse
from pong_decision_transformer.train import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="mujoco/halfcheetah/medium-v0", help="Minari Environment")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--context-length", type=int, default=40)
    args = parser.parse_args()

    train(
        env=args.env,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        context_length=args.context_length
    )

if __name__ == "__main__":
    main()
