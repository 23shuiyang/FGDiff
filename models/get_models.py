import copy
import logging

from models.backbone import *
from ptflops import get_model_complexity_info

def get_models(args):
    # get student model
    if args.student == "eeeac2":
        student = EEEAC2(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size,
                         readout=args.readout, decoder=args.decoder, is_f=args.is_f)
    elif args.student == "mbv3":
        student = MobileNetV3_1k(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size,
                                 readout=args.readout, decoder=args.decoder, is_f=args.is_f)
    elif args.student == "efb0":
        student = EfficientNet(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size,
                               readout=args.readout, decoder=args.decoder, is_f=args.is_f)
    elif args.student == "efb4":
        student = EfficientNetB4(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size,
                                 readout=args.readout, decoder=args.decoder, is_f=args.is_f)
    elif args.student == "repvit15":
        student = RepViT_m1_5(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size,
                                 readout=args.readout, decoder=args.decoder, is_f=args.is_f)
    elif args.student == "repvit09":
        student = RepViT_m0_9(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size,
                                 readout=args.readout, decoder=args.decoder, is_f=args.is_f)
    elif args.student == "ofa595":
        student = OFA595(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size,
                            readout=args.readout, decoder=args.decoder, is_f=args.is_f)
    elif args.student == "efv2s2":
        student = EfficientFormerV2_S2(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size,
                                 readout=args.readout, decoder=args.decoder, is_f=args.is_f)

    # get teacher model
    if args.teacher == "ofa595":
        teacher = OFA595(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size,
                             readout=args.readout, decoder=args.decoder, is_f=args.is_f)
    elif args.teacher == "efb0":
        teacher = EfficientNet(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size,
                                   readout=args.readout, decoder=args.decoder, is_f=args.is_f)
    elif args.teacher == "repvit15":
        teacher = RepViT_m1_5(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size,
                                 readout=args.readout, decoder=args.decoder, is_f=args.is_f)
    elif args.teacher == "efb4":
        teacher = EfficientNetB4(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size,
                                     readout=args.readout, decoder=args.decoder, is_f=args.is_f)
    elif args.teacher == "efb7":
        teacher = EfficientNetB7(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size,
                                     readout=args.readout, decoder=args.decoder, is_f=args.is_f)

    if args.mode == 'ps-kd':
        teacher = copy.deepcopy(student)
    return student, teacher

def get_model_performance(args, teacher, student):
    if args.mode == "baseline" or args.mode == "ps-kd" or args.mode == "self-kd" or args.mode == "ema-kd" or args.mode == "dda-skd":
        logging.info('{:<30}'.format('Teacher: None'))
        macs_t, params_t = 0, 0
        logging.info('{:<30}  {:<8}'.format('Student: ', args.student))
        print('{:<30}  {:<8}'.format('Student: ', args.student))
        macs_s, params_s = get_model_complexity_info(student, (3, args.input_size_h, args.input_size_w), as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        logging.info('{:<30}  {:<8}'.format('Computational complexity: ', macs_s))
        logging.info('{:<30}  {:<8}'.format('Number of parameters: ', params_s))
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs_s))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params_s))
        return macs_t, params_t, macs_s, params_s
    else:
        logging.info('{:<30}  {:<8}'.format('Teacher: ', args.teacher))
        macs_t, params_t = get_model_complexity_info(teacher, (3, args.input_size_h, args.input_size_w), as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        logging.info('{:<30}  {:<8}'.format('Computational complexity: ', macs_t))
        logging.info('{:<30}  {:<8}'.format('Number of parameters: ', params_t))
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs_t))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params_t))
        logging.info('{:<30}  {:<8}'.format('Student: ', args.student))
        print('{:<30}  {:<8}'.format('Student: ', args.student))
        macs_s, params_s = get_model_complexity_info(student, (3, args.input_size_h, args.input_size_w), as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        logging.info('{:<30}  {:<8}'.format('Computational complexity: ', macs_s))
        logging.info('{:<30}  {:<8}'.format('Number of parameters: ', params_s))
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs_s))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params_s))
        return macs_t, params_t, macs_s, params_s
