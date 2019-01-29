import argparse
import todo, img_aug

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--todo", default='train', type = str, help="train, test, img_aug")
    parser.add_argument("--input_shape", default=[256, 256, 1], type=list)
    parser.add_argument("--label_shape", default=[256, 256, 1], type=list)
    parser.add_argument("--batch_size", default=16, type = int)
    parser.add_argument("--epoch", default=100, type = int)
    parser.add_argument("--model_num", default="1000", type = str)
    parser.add_argument("--drop_out", default="False", type = str)

    parser.add_argument("--save_model_rate", default=50, type = int)
    parser.add_argument("--aug_size", default=30, type = int)
    ##############################################
    args = parser.parse_args()
    ##############################################
    if args.todo == "train": todo.train(args)
    if args.todo == "test": todo.test(args)
    if args.todo == "img_aug": img_aug.aug(args)

if __name__ == "__main__":
    main()