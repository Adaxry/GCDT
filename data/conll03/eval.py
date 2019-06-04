import sys
import os
import argparse

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Modify the file format for conlleval.py')
    parser.add_argument('-src_file', '-s',
                        help='src file path')
    parser.add_argument('-predict_file', '-p',
                        help='predict file path')
    parser.add_argument('-golden_file', '-g',
                        help='golden file path')
    parser.add_argument('-result_file', '-r',
                        help='Modified file path')
    return parser.parse_args(args)


def main(args):
    with open(args.src_file, "r", encoding="utf-8") as src_f, \
         open(args.predict_file, "r", encoding="utf-8") as pred_f, \
         open(args.golden_file, "r", encoding="utf-8") as gold_f, \
         open(args.result_file, "w", encoding="utf-8") as res_f:
        bad_pred_num = 0
        for src, pred, gold in zip(src_f, pred_f, gold_f):
            src_words = src.strip().split()
            gold_labels = gold.strip().split()
            pred_labels = pred.strip().split()
            if len(gold_labels) != len(pred_labels):
                bad_pred_num += 1
                #print(gold_labels)
                #print(pred_labels)
                continue
            for i in range(len(src_words)):
                new_line = src_words[i] + " " + gold_labels[i] + " " + pred_labels[i] + "\n"
                res_f.write(new_line)
            res_f.write("\n")
        print("Get {} bad pred".format(bad_pred_num))
    os.system("./conlleval < {}".format(args.result_file))
    os.remove(args.result_file)    


if __name__ == '__main__':
    main(parse_args())



