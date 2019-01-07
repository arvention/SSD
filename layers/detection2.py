import torch
from torch.autograd import Function
from utils.bbox_utils import decode


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, class_count):
        self.class_count = class_count
        self.variance = [0.1, 0.2]

    def forward(self, conf_data, loc_data, prior):

        loc_data = loc_data.data
        conf_data = conf_data.data
        prior_data = prior.data
        num = loc_data.size(0)  # batch size

        self.num_priors = prior_data.size(0)
        self.boxes = torch.zeros(num, self.num_priors, 4)
        self.scores = torch.zeros(num, self.num_priors, self.num_classes)

        if num == 1:
            # size batch x num_classes x num_priors
            conf_preds = conf_data.unsqueeze(0)

        else:
            conf_preds = conf_data.view(num, self.num_priors,
                                        self.num_classes)
            self.boxes.expand(num, self.num_priors, 4)
            self.scores.expand(num, self.num_priors, self.num_classes)

        # Decode predictions into bboxes.
        for i in range(num):
            default = prior_data
            decoded_boxes = decode(loc_data[i], default, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            self.boxes[i] = decoded_boxes
            self.scores[i] = conf_scores

        return self.boxes, self.scores
