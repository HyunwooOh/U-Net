import argparse
import todo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--todo", default='train', type = str, help="train, test")
    parser.add_argument("--input_shape", default=[256, 256, 1], type=list)
    parser.add_argument("--label_shape", default=[256, 256, 1], type=list)
    parser.add_argument("--batch_size", default=32, type = int)
    parser.add_argument("--epoch", default=1000, type = int)
    parser.add_argument("--model_num", default=str(1000), type = str)
    parser.add_argument("--save_model_rate", default=100, type = int)
    ##############################################
    args = parser.parse_args()
    ##############################################
    if args.todo == "train": todo.train(args)
    if args.todo == "test": todo.test(args)

if __name__ == "__main__":
    main()