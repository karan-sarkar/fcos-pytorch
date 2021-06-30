import torch
from torch import nn


INF = 100000000


class IOULoss(nn.Module):
    def __init__(self, loc_loss_type):
        super().__init__()

        self.loc_loss_type = loc_loss_type

    def forward(self, out, target, weight=None):
        pred_left, pred_top, pred_right, pred_bottom = out.unbind(1)
        target_left, target_top, target_right, target_bottom = target.unbind(1)

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(
            pred_right, target_right
        )
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(
            pred_top, target_top
        )

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        ious = (area_intersect + 1) / (area_union + 1)

        if self.loc_loss_type == 'iou':
            loss = -torch.log(ious)

        elif self.loc_loss_type == 'giou':
            g_w_intersect = torch.max(pred_left, target_left) + torch.max(
                pred_right, target_right
            )
            g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(
                pred_top, target_top
            )
            g_intersect = g_w_intersect * g_h_intersect + 1e-7
            gious = ious - (g_intersect - area_union) / g_intersect

            loss = 1 - gious

        if weight is not None and weight.sum() > 0:
            return (loss * weight).sum() / weight.sum()

        else:
            return loss.mean()


def clip_sigmoid(input):
    out = torch.clamp(torch.sigmoid(input), min=1e-4, max=1 - 1e-4)

    return out


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha

    def compute(self, out, target):
        n_class = out.shape[1]
        class_ids = torch.arange(
            n_class, dtype=target.dtype, device=target.device
        ).unsqueeze(0)

        t = target.unsqueeze(1)
        p = torch.softmax(out, 1)
        
        gamma = self.gamma
        alpha = self.alpha

        term1 = (1 - p) ** gamma * torch.log(p)
        term2 = p ** gamma * torch.log(1 - p)
        loss = (
            -(t == class_ids).float() * alpha * term1
            - ((t != class_ids) * (t >= 0)).float() * (1 - alpha) * term2
        )

        return loss
    
    def forward(self, predicted_scores, true_classes):
        batch_size, n_priors, n_classes = predicted_scores.shape
        
        positive_priors = true_classes != 0
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = 3 * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.compute(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors, n_classes).sum(-1)  # (N, 8732)
        
        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(true_classes.device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) # (), scalar
        return conf_loss
         
         


class FCOSLoss(nn.Module):
    def __init__(
        self, sizes, gamma, alpha, iou_loss_type, center_sample, fpn_strides, pos_radius
    ):
        super().__init__()

        self.sizes = sizes

        self.cls_loss = SigmoidFocalLoss(gamma, alpha)
        self.box_loss = IOULoss(iou_loss_type)
        self.center_loss = nn.BCEWithLogitsLoss()

        self.center_sample = center_sample
        self.strides = fpn_strides
        self.radius = pos_radius

    def prepare_target(self, points, targets):
        ex_size_of_interest = []

        for i, point_per_level in enumerate(points):
            size_of_interest_per_level = point_per_level.new_tensor(self.sizes[i])
            ex_size_of_interest.append(
                size_of_interest_per_level[None].expand(len(point_per_level), -1)
            )

        ex_size_of_interest = torch.cat(ex_size_of_interest, 0)
        n_point_per_level = [len(point_per_level) for point_per_level in points]
        point_all = torch.cat(points, dim=0)
        label, box_target = self.compute_target_for_location(
            point_all, targets, ex_size_of_interest, n_point_per_level
        )

        for i in range(len(label)):
            label[i] = torch.split(label[i], n_point_per_level, 0)
            box_target[i] = torch.split(box_target[i], n_point_per_level, 0)

        label_level_first = []
        box_target_level_first = []

        for level in range(len(points)):
            label_level_first.append(
                torch.cat([label_per_img[level] for label_per_img in label], 0)
            )
            box_target_level_first.append(
                torch.cat(
                    [box_target_per_img[level] for box_target_per_img in box_target], 0
                )
            )

        return label_level_first, box_target_level_first

    def get_sample_region(self, gt, strides, n_point_per_level, xs, ys, radius=1):
        n_gt = gt.shape[0]
        n_loc = len(xs)
        gt = gt[None].expand(n_loc, n_gt, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2

        if center_x[..., 0].sum() == 0:
            return xs.new_zeros(xs.shape, dtype=torch.uint8)

        begin = 0

        center_gt = gt.new_zeros(gt.shape)

        for level, n_p in enumerate(n_point_per_level):
            end = begin + n_p
            stride = strides[level] * radius

            x_min = center_x[begin:end] - stride
            y_min = center_y[begin:end] - stride
            x_max = center_x[begin:end] + stride
            y_max = center_y[begin:end] + stride

            center_gt[begin:end, :, 0] = torch.where(
                x_min > gt[begin:end, :, 0], x_min, gt[begin:end, :, 0]
            )
            center_gt[begin:end, :, 1] = torch.where(
                y_min > gt[begin:end, :, 1], y_min, gt[begin:end, :, 1]
            )
            center_gt[begin:end, :, 2] = torch.where(
                x_max > gt[begin:end, :, 2], gt[begin:end, :, 2], x_max
            )
            center_gt[begin:end, :, 3] = torch.where(
                y_max > gt[begin:end, :, 3], gt[begin:end, :, 3], y_max
            )

            begin = end

        left = xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - xs[:, None]
        top = ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - ys[:, None]

        center_bbox = torch.stack((left, top, right, bottom), -1)
        is_in_boxes = center_bbox.min(-1)[0] > 0

        return is_in_boxes

    def compute_target_for_location(
        self, locations, targets, sizes_of_interest, n_point_per_level
    ):
        labels = []
        box_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for i in range(len(targets)):
            targets_per_img = targets[i]
            assert targets_per_img.mode == 'xyxy'
            bboxes = targets_per_img.box
            labels_per_img = targets_per_img.fields['labels']
            area = targets_per_img.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]

            box_targets_per_img = torch.stack([l, t, r, b], 2)

            if self.center_sample:
                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, n_point_per_level, xs, ys, radius=self.radius
                )

            else:
                is_in_boxes = box_targets_per_img.min(2)[0] > 0

            max_box_targets_per_img = box_targets_per_img.max(2)[0]

            is_cared_in_level = (
                max_box_targets_per_img >= sizes_of_interest[:, [0]]
            ) & (max_box_targets_per_img <= sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_level == 0] = INF

            locations_to_min_area, locations_to_gt_id = locations_to_gt_area.min(1)

            box_targets_per_img = box_targets_per_img[
                range(len(locations)), locations_to_gt_id
            ]
            labels_per_img = labels_per_img[locations_to_gt_id]
            labels_per_img[locations_to_min_area == INF] = 0

            labels.append(labels_per_img)
            box_targets.append(box_targets_per_img)

        return labels, box_targets

    def compute_centerness_targets(self, box_targets):
        left_right = box_targets[:, [0, 2]]
        top_bottom = box_targets[:, [1, 3]]
        centerness = (left_right.min(-1)[0] / left_right.max(-1)[0]) * (
            top_bottom.min(-1)[0] / top_bottom.max(-1)[0]
        )

        return torch.sqrt(centerness)

    def forward(self, locations, cls_pred, box_pred, center_pred, targets):
        batch = cls_pred[0].shape[0]
        n_class = cls_pred[0].shape[1]

        labels, box_targets = self.prepare_target(locations, targets)

        cls_flat = []
        box_flat = []
        center_flat = []

        labels_flat = []
        labels_flat2 = []
        box_targets_flat = []

        for i in range(len(labels)):
            cls_flat.append(cls_pred[i].permute(0, 2, 3, 1).reshape(cls_pred[i].shape[0], -1, n_class))
            box_flat.append(box_pred[i].permute(0, 2, 3, 1).reshape(-1, 4))
            center_flat.append(center_pred[i].permute(0, 2, 3, 1).reshape(-1))

            labels_flat.append(labels[i].reshape(cls_pred[i].shape[0], -1))
            labels_flat2.append(labels[i].reshape(-1))
            box_targets_flat.append(box_targets[i].reshape(-1, 4))

        cls_flat = torch.cat(cls_flat, 1)
        box_flat = torch.cat(box_flat, 0)
        center_flat = torch.cat(center_flat, 0)

        labels_flat = torch.cat(labels_flat, 1)
        labels_flat2 = torch.cat(labels_flat2, 0)
        box_targets_flat = torch.cat(box_targets_flat, 0)

        pos_id = torch.nonzero(labels_flat2 > 0).squeeze(1)

        cls_loss = self.cls_loss(cls_flat, labels_flat.int()) / (pos_id.numel() + batch)

        box_flat = box_flat[pos_id]
        center_flat = center_flat[pos_id]

        box_targets_flat = box_targets_flat[pos_id]

        if pos_id.numel() > 0:
            center_targets = self.compute_centerness_targets(box_targets_flat)

            box_loss = self.box_loss(box_flat, box_targets_flat, center_targets)
            center_loss = self.center_loss(center_flat, center_targets)

        else:
            box_loss = box_flat.sum()
            center_loss = center_flat.sum()

        return cls_loss, box_loss, center_loss
