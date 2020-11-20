from .bert import BERTTrainer
from .dae import DAETrainer
from .vae import VAETrainer

#from trainers.bert import BERTTrainer
#from trainers.dae import DAETrainer
#from trainers.vae import VAETrainer


TRAINERS = {
    BERTTrainer.code(): BERTTrainer,
    DAETrainer.code(): DAETrainer,
    VAETrainer.code(): VAETrainer
}


def trainer_factory(args, model, train_loader, val_loader, test_loader, export_root):
    print("args.trainer_code:", args.trainer_code)
    print("TRAINERS:", TRAINERS)
    trainer = TRAINERS[args.trainer_code]
    return trainer(args, model, train_loader, val_loader, test_loader, export_root)
