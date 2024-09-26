CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "log"

IGNORE_INDEX = -100
# DEFAULT_PAD_TOKEN = "[PAD]"

DEFAULT_PAD_TOKEN = "<|endoftext|>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_BOX_TOKEN = "<box>"

DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'

DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'



CONVERSATION_DATA = {

    'doclaynet_train_data': {
        'images': '/home/t-jiaweiwang/Project/GOT-OCR2.0/datasets/DocLayNet/',
        'annotations': '/home/t-jiaweiwang/Project/GOT-OCR2.0/datasets/DocLayNet/got_annotations/train_got_conversations.json',
    },
    'doclaynet_train_data_wo_others': {
        'images': '/home/t-jiaweiwang/Project/GOT-OCR2.0/datasets/DocLayNet/',
        'annotations': '/home/t-jiaweiwang/Project/GOT-OCR2.0/datasets/DocLayNet/got_annotations/train_got_conversations_wo_others.json',
    },
    # 'data_2': {
    #     'images': '/path/',
    #     'annotations': '/path/data2.json',
    # },
    # 'data_3': {
    #     'images': '/path/',
    #     'annotations': '/path/data3.json',
    # },


}