import train
import pretrain
import inference


if __name__ == "__main__":
    pretrain.pretrain("input/train2.csv","input/train/")
    #train.train()
    #inference.predict()