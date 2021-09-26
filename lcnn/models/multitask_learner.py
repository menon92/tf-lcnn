import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from . hourglass_pose import hourglass_net

def multitask_head(x, num_class=5, name=None):
    '''Multitaks head implimentaiton
    Args:
        x (Tensor): input to multitaks head
        num_class (int): number of class or task
    '''

    head_size = [2, 1, 2]
    print(f"multitask_head input shape::{K.int_shape(x)}")
    m_out = K.int_shape(x)[-1] // 4
    print(f'm_out {m_out}')

    # head size 2
    x2 = layers.Conv2D(m_out, kernel_size=3, padding='SAME')(x)
    x2 = layers.Activation('relu')(x2)
    x2 = layers.Conv2D(head_size[0], kernel_size=1, padding='SAME')(x2)

    # head size 1
    x1 = layers.Conv2D(m_out, kernel_size=3, padding='SAME')(x)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Conv2D(head_size[1], kernel_size=1, padding='SAME')(x1)
 
    # head size 2
    x3 = layers.Conv2D(m_out, kernel_size=3, padding='SAME')(x)
    x3 = layers.Activation('relu')(x3)
    x3 = layers.Conv2D(head_size[2], kernel_size=1, padding='SAME')(x3)

    # combine three head
    out = layers.Concatenate(axis=-1, name=name)([x2, x1, x3])

    return out


def l2loss(input, target):
    return ((target - input) ** 2).mean(2).mean(1)


def cross_entropy_loss(logits, positive):
    nlogp = -F.log_softmax(logits, dim=0)
    return (positive * nlogp[1] + (1 - positive) * nlogp[0]).mean(2).mean(1)


def sigmoid_l1_loss(logits, target, offset=0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp - target)
    if mask is not None:
        w = mask.mean(2, True).mean(1, True)
        w[w == 0] = 1
        loss = loss * (mask / w)

    return loss.mean(2).mean(1)



def multitask_learner(input_dict):
    num_class = 5
    head_size = np.array([2, 1, 2])

    image = input_dict['image']
    outputs, feature = hourglass_net(image)
    result = {"feature": feature}
    batch, channel, row, col = outputs[0].shape

    T = input_dict["target"].copy()
    n_jtyp = T["jmap"].shape[1]

    # switch to CNHW
    for task in ["jmap"]:
        T[task] = T[task].permute(1, 0, 2, 3)
    for task in ["joff"]:
        T[task] = T[task].permute(1, 2, 0, 3, 4)

    offset = [2, 3, 5]
    loss_weight = M.loss_weight
    losses = []
    for stack, output in enumerate(outputs):
        output = output.transpose(0, 1).reshape([-1, batch, row, col]) # .contiguous()
        jmap = output[0 : offset[0]].reshape(n_jtyp, 2, batch, row, col)
        lmap = output[offset[0] : offset[1]].squeeze(0)
        joff = output[offset[1] : offset[2]].reshape(n_jtyp, 2, batch, row, col)
        if stack == 0:
            result["preds"] = {
                "jmap": jmap.permute(2, 0, 1, 3, 4).softmax(2)[:, :, 1],
                "lmap": lmap.sigmoid(),
                "joff": joff.permute(2, 0, 1, 3, 4).sigmoid() - 0.5,
            }
            if input_dict["mode"] == "testing":
                return result

        L = OrderedDict()
        L["jmap"] = sum(
            cross_entropy_loss(jmap[i], T["jmap"][i]) for i in range(n_jtyp)
        )
        L["lmap"] = (
            F.binary_cross_entropy_with_logits(lmap, T["lmap"], reduction="none")
            .mean(2)
            .mean(1)
        )
        L["joff"] = sum(
            sigmoid_l1_loss(joff[i, j], T["joff"][i, j], -0.5, T["jmap"][i])
            for i in range(n_jtyp)
            for j in range(2)
        )
        for loss_name in L:
            L[loss_name].mul_(loss_weight[loss_name])
        losses.append(L)
    result["losses"] = losses

    return result


def l2loss(input, target):
    return ((target - input) ** 2).mean(2).mean(1)


def cross_entropy_loss(logits, positive):
    nlogp = tf.nn.log_softmax(logits, dim=0)
    return (positive * nlogp[1] + (1 - positive) * nlogp[0]).mean(2).mean(1)


def sigmoid_l1_loss(logits, target, offset=0.0, mask=None):
    logp = tf.math.sigmoid(logits) + offset
    loss = tf.math.abs(logp - target)
    if mask is not None:
        w = mask.mean(2, True).mean(1, True)
        # w[w == 0] = 1
        condition = tf.equal(w, 0)
        w = tf.where(condition, 1.0, w)
        
        loss = loss * (mask / w)

    return loss.mean(2).mean(1)
