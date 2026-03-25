import logging

import numpy as np

import utils
import torch
from utils import loss_func, loss_func_features
import time
import sys
def train_baseline(student, optimizer, loader, epoch, device, args, scaler):
    student.train()
    tic = time.time()
    total_loss = 0.0
    cur_loss = 0.0
    for idx, (img, gt, fixations) in enumerate(loader):
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.amp):
            pred_map_student = student(img)
            loss = utils.loss_func(pred_map_student, gt, fixations, args)
        scaler.scale(loss).backward()
        total_loss += loss.item()
        cur_loss += loss.item()
        scaler.step(optimizer)
        scaler.update()
        if idx % args.log_interval == (args.log_interval - 1):
            logging.info('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes, GPU Mem: {:.2f} MB'.format(epoch, idx,
                                                                                                           cur_loss / args.log_interval,
                                                                                                           (
                                                                                                                       time.time() - tic) / 60,
                                                                                                           round(
                                                                                                               torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)))
            cur_loss = 0.0
            sys.stdout.flush()
    logging.info('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss / len(loader)))
    sys.stdout.flush()
    return total_loss / len(loader)


def train_pkd(student, optimizer, loader, epoch, device, args, scaler, teacher):
    student.train()
    teacher.train()
    tic = time.time()
    total_loss = 0.0
    cur_loss = 0.0
    for idx, (img, gt, fixations) in enumerate(loader):
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.amp):
            pred_map = teacher(img)
            loss = utils.loss_func(pred_map, gt, fixations, args)
        with torch.cuda.amp.autocast(enabled=args.amp):
            pred_map_student = student(img)
            loss_s = loss_func(pred_map_student, pred_map.detach(), fixations, args)
        scaler.scale(loss + loss_s).backward()
        total_loss += loss.item() + loss_s.item()
        cur_loss += loss.item() + loss_s.item()
        scaler.step(optimizer)
        scaler.update()
        if idx % args.log_interval == (args.log_interval - 1):
            logging.info('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes, GPU Mem: {:.2f} MB'.format(epoch, idx,
                                                                                                           cur_loss / args.log_interval,
                                                                                                           (
                                                                                                                       time.time() - tic) / 60,
                                                                                                           round(
                                                                                                               torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)))
            cur_loss = 0.0
            sys.stdout.flush()
    logging.info('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss / len(loader)))
    sys.stdout.flush()
    return total_loss / len(loader)


def train_ps_kd(student, optimizer, loader, epoch, device, args, scaler, teacher):
    student.train()
    teacher.eval()
    tic = time.time()
    total_loss = 0.0
    cur_loss = 0.0
    alphaT = 0.8
    alpha_t = alphaT * ((epoch + 1) / args.no_epochs)
    alpha_t = max(0, alpha_t)
    for idx, (img, gt, fixations) in enumerate(loader):
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)
        optimizer.zero_grad()
        if epoch > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=args.amp):
                    pred_map_teacher = teacher(img)
                    soft_logits = pred_map_teacher.clone().detach()
                    # update gt by PS-KD
                    gt = (alpha_t * soft_logits) + ((1-alpha_t) * gt)
                    loss = loss_func(pred_map_teacher, gt, fixations, args)
        with torch.cuda.amp.autocast(enabled=args.amp):
            pred_map_student = student(img)
            loss_s = loss_func(pred_map_student, gt, fixations, args)
        if epoch > 0:
            scaler.scale(loss + loss_s).backward()
            total_loss += loss.item() + loss_s.item()
            cur_loss += loss.item() + loss_s.item()
        else:
            scaler.scale(loss_s).backward()
            total_loss += loss_s.item()
            cur_loss += loss_s.item()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        if idx % args.log_interval == (args.log_interval - 1):
            logging.info('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes, GPU Mem: {:.2f} MB'.format(epoch, idx,
                                                                                                           cur_loss / args.log_interval,
                                                                                                           (
                                                                                                                       time.time() - tic) / 60,
                                                                                                           round(
                                                                                                               torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)))

            cur_loss = 0.0
            sys.stdout.flush()
    logging.info('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss / len(loader)))
    sys.stdout.flush()
    return total_loss / len(loader)

def train_self_kd(student, optimizer, loader, epoch, device, args, scaler, swa_model):
    student.train()
    tic = time.time()
    total_loss = 0.0
    cur_loss = 0.0
    for idx, (img, gt, fixations) in enumerate(loader):
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)
        optimizer.zero_grad()
        if epoch > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=args.amp):
                    pred_map = swa_model(img)
                    soft_logits = pred_map.clone().detach()
                    gt = soft_logits + (gt - soft_logits) * 0.5
                    loss = loss_func(pred_map, gt, fixations, args)
        with torch.cuda.amp.autocast(enabled=args.amp):
            pred_map_student = student(img)
            loss_s = loss_func(pred_map_student, gt, fixations, args)
        if epoch > 0:
            scaler.scale(loss + loss_s).backward()
            total_loss += loss.item() + loss_s.item()
            cur_loss += loss.item() + loss_s.item()
        else:
            scaler.scale(loss_s).backward()
            total_loss += loss_s.item()
            cur_loss += loss_s.item()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        if idx % args.log_interval == (args.log_interval - 1):
            logging.info('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes, GPU Mem: {:.2f} MB'.format(epoch, idx,
                                                                                                           cur_loss / args.log_interval,
                                                                                                           (
                                                                                                                       time.time() - tic) / 60,
                                                                                                           round(
                                                                                                               torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)))
            cur_loss = 0.0
            sys.stdout.flush()
    logging.info('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss / len(loader)))
    sys.stdout.flush()
    return total_loss / len(loader)

def train_ema_kd(student, optimizer, loader, epoch, device, args, scaler, ema_model):
    student.train()
    tic = time.time()
    total_loss = 0.0
    cur_loss = 0.0
    for idx, (img, gt, fixations) in enumerate(loader):
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.amp):
            if epoch > 0:
                with torch.no_grad():
                    pred_map_ema = ema_model(img)
                    soft_logits_ema = pred_map_ema.clone().detach()
                    gt = soft_logits_ema + (gt - soft_logits_ema) * 0.5
            pred_map_student = student(img)
            loss = loss_func(pred_map_student, gt, fixations, args)
        scaler.scale(loss).backward()
        total_loss += loss.item()
        cur_loss += loss.item()
        scaler.step(optimizer)
        scaler.update()
        if epoch > 0:
            ema_model.update_parameters(student)
        if idx % args.log_interval == (args.log_interval - 1):
            logging.info('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes, GPU Mem: {:.2f} MB'.format(epoch, idx,
                                                                                                           cur_loss / args.log_interval,
                                                                                                           (
                                                                                                                       time.time() - tic) / 60,
                                                                                                           round(
                                                                                                               torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)))
            cur_loss = 0.0
            sys.stdout.flush()
    logging.info('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss / len(loader)))
    sys.stdout.flush()
    return total_loss / len(loader)

def train_dda_skd(student, optimizer, loader, epoch, device, args, scaler, ema_model, swa_model):
    student.train()
    tic = time.time()
    total_loss = 0.0
    cur_loss = 0.0
    k = args.k
    x_t = (epoch + 1) / args.no_epochs
    x_T = 1 / (1+np.exp(-1*k*(x_t-0.5)))
    for idx, (img, gt, fixations) in enumerate(loader):
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.amp):
            if epoch > 0:
                with torch.no_grad():
                    pred_map_ema = ema_model(img)
                    soft_logits_ema = pred_map_ema.clone().detach()
                    pred_map_swa = swa_model(img)
                    soft_logits_swa = pred_map_swa.clone().detach()
                    pseudo = soft_logits_ema*(1-x_T) + soft_logits_swa*x_T
                pred_map_student = student(img)
                loss_1 = loss_func(pred_map_student, pseudo, fixations, args)
                loss_2 = loss_func(pred_map_student, gt, fixations, args)
                loss_s = loss_1 + loss_2
            else:
                pred_map_student = student(img)
                loss_s = loss_func(pred_map_student, gt, fixations, args)
        scaler.scale(loss_s).backward()
        total_loss += loss_s.item()
        cur_loss += loss_s.item()
        scaler.step(optimizer)
        scaler.update()
        if 0 < epoch:
            ema_model.update_parameters(student)

        if idx % args.log_interval == (args.log_interval - 1):
            logging.info('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes, GPU Mem: {:.2f} MB'.format(epoch, idx,
                                                                                                           cur_loss / args.log_interval,
                                                                                                           (
                                                                                                                       time.time() - tic) / 60,
                                                                                                           round(
                                                                                                               torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)))
            cur_loss = 0.0
            sys.stdout.flush()
    logging.info('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss / len(loader)))
    sys.stdout.flush()
    return total_loss / len(loader)
def train_pkd_skd(student, optimizer, loader, epoch, device, args, scaler, ema_model, swa_model, teacher):
    student.train()
    teacher.eval()
    tic = time.time()
    total_loss = 0.0
    cur_loss = 0.0
    a = args.a
    k = args.k
    x_t = (epoch + 1) / args.no_epochs
    x_T = 1 / (1 + np.exp(-1 * k * (x_t - 0.5)))
    for idx, (img, gt, fixations) in enumerate(loader):
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.amp):
            with torch.no_grad():
                features_t, pred_t = teacher(img)
                pred_t = (pred_t + gt)*0.5
            if epoch > 0:
                with torch.no_grad():
                    _, pred_ema = ema_model(img)
                    soft_logits_ema = pred_ema.clone().detach()
                    _, pred_swa = swa_model(img)
                    soft_logits_swa = pred_swa.clone().detach()
                    pseudo = soft_logits_ema*(1-x_T) + soft_logits_swa*x_T
                features_s, pred_s = student(img)
                loss_1 = loss_func(pred_s, pseudo, fixations, args)
                loss_2 = loss_func(pred_s, pred_t, fixations, args)
                loss_s = loss_1 + loss_2
            else:
                features_s, pred_s = student(img)
                loss_s = loss_func(pred_s, pred_t, fixations, args)
            loss_f0 = loss_func_features(features_s[0], features_t[0], args)
            loss_f1 = loss_func_features(features_s[1], features_t[1], args)
            loss_f2 = loss_func_features(features_s[2], features_t[2], args)
            loss_f3 = loss_func_features(features_s[3], features_t[3], args)
            loss_f4 = loss_func_features(features_s[4], features_t[4], args)
            loss_f = loss_f0 * args.f0 + loss_f1 * args.f1 + loss_f2 * args.f2 + loss_f3 * args.f3 + loss_f4 * args.f4

            loss = loss_s * (1-a) + loss_f * a
        scaler.scale(loss).backward()
        total_loss += loss.item()
        cur_loss += loss.item()
        scaler.step(optimizer)
        scaler.update()
        if 0 < epoch:
            ema_model.update_parameters(student)
        if idx % args.log_interval == (args.log_interval - 1):
            logging.info('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes, GPU Mem: {:.2f} MB'.format(epoch, idx,
                                                                                                           cur_loss / args.log_interval,
                                                                                                           (
                                                                                                                   time.time() - tic) / 60,
                                                                                                           round(
                                                                                                               torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)))
            cur_loss = 0.0
            sys.stdout.flush()
    logging.info('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss / len(loader)))
    sys.stdout.flush()
    return total_loss / len(loader)

def train_pkd_skd_two(student, optimizer, loader, epoch, device, args, scaler, ema_model, swa_model, teacher1, teacher2):
    student.train()
    teacher1.eval()
    teacher2.eval()
    tic = time.time()
    total_loss = 0.0
    cur_loss = 0.0
    a = args.a
    k = args.k
    x_t = (epoch + 1) / args.no_epochs
    x_T = 1 / (1 + np.exp(-1 * k * (x_t - 0.5)))
    for idx, (img, gt, fixations) in enumerate(loader):
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.amp):
            with torch.no_grad():
                features_t1, pred_t1 = teacher1(img)
                features_t2, pred_t2 = teacher2(img)
                pred_t = (pred_t1 + pred_t2)*0.5
                pred_t = (pred_t + gt)*0.5
            if epoch > 0:
                with torch.no_grad():
                    _, pred_ema = ema_model(img)
                    soft_logits_ema = pred_ema.clone().detach()
                    _, pred_swa = swa_model(img)
                    soft_logits_swa = pred_swa.clone().detach()
                    pseudo = soft_logits_ema*(1-x_T) + soft_logits_swa*x_T
                features_s, pred_s = student(img)
                loss_1 = loss_func(pred_s, pseudo, fixations, args)
                loss_2 = loss_func(pred_s, pred_t, fixations, args)
                loss_s = loss_1 + loss_2
            else:
                features_s, pred_s = student(img)
                loss_s = loss_func(pred_s, pred_t, fixations, args)
            loss_f0 = loss_func_features(features_s[0], (features_t1[0]+features_t2[0])*0.5, args)
            loss_f1 = loss_func_features(features_s[1], (features_t1[1]+features_t2[1])*0.5, args)
            loss_f2 = loss_func_features(features_s[2], (features_t1[2]+features_t2[2])*0.5, args)
            loss_f3 = loss_func_features(features_s[3], (features_t1[3]+features_t2[3])*0.5, args)
            loss_f4 = loss_func_features(features_s[4], (features_t1[4]+features_t2[4])*0.5, args)
            loss_f = loss_f0 * args.f0 + loss_f1 * args.f1 + loss_f2 * args.f2 + loss_f3 * args.f3 + loss_f4 * args.f4

            loss = loss_s * (1-a) + loss_f * a
        scaler.scale(loss).backward()
        total_loss += loss.item()
        cur_loss += loss.item()
        scaler.step(optimizer)
        scaler.update()
        if 0 < epoch:
            ema_model.update_parameters(student)
        if idx % args.log_interval == (args.log_interval - 1):
            logging.info('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes, GPU Mem: {:.2f} MB'.format(epoch, idx,
                                                                                                           cur_loss / args.log_interval,
                                                                                                           (
                                                                                                                   time.time() - tic) / 60,
                                                                                                           round(
                                                                                                               torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)))
            cur_loss = 0.0
            sys.stdout.flush()
    logging.info('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss / len(loader)))
    sys.stdout.flush()
    return total_loss / len(loader)