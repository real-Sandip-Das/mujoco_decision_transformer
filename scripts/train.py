import argparse
from accelerate import notebook_launcher

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--env", type=str, default="mujoco/halfcheetah/medium-v0", help="Minari Environment")
#     parser.add_argument("--batch-size", type=int, default=4096)
#     parser.add_argument("--epochs", type=int, default=10000)
#     parser.add_argument("--lr", type=float, default=1e-2)
#     parser.add_argument("--context-length", type=int, default=40)
#     args = parser.parse_args()

#     notebook_launcher(train(
#         env=args.env,
#         batch_size=args.batch_size,
#         epochs=args.epochs,
#         lr=args.lr,
#         context_length=args.context_length
#     ), num_processes=1)

# if __name__ == "__main__":
#     main()
def wrapper():
    from pong_decision_transformer.train import train
    train("mujoco/halfcheetah/medium-v0",1024,10000,1e-2)
notebook_launcher(wrapper,num_processes=1)
