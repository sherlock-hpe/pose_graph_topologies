import torch
import torch.nn.functional as F
from networkx.classes import edges
from torch import dtype
from torch_geometric.data import Data, Batch
from typing import Tuple

def build_keypoints_adj_matrix(inputs: torch.Tensor, data_type, device):

    if data_type == 'taiji' :
        #taiji
        edge_key = [[0, 2], [0, 22], [1, 2], [21, 22], [1, 23], [21, 23],
                    [2, 3], [22, 3], [2, 4], [22, 4], [4, 5], [4, 24],
                    [5, 6], [24, 25], [5, 7], [24, 26], [6, 7], [25, 26],
                    [6, 8], [25, 27], [8, 12], [27, 28], [12, 13], [28, 31],
                    [13, 14], [13, 15], [31, 32], [31, 33], [7, 9], [26, 39],
                    [9, 39], [9, 10], [39, 40], [10, 40], [10, 11], [29, 40],
                    [11, 30], [29, 30], [4, 30], [11, 16], [29, 34], [16, 17],
                    [34, 35], [17, 18], [17, 19], [17, 20], [35, 36], [35, 37],
                    [35, 38]]
    elif data_type == 'mpii' :
        edge_key = [
            (1, 0),  #  Right Knee-Right Ankle
            (2, 1),  #  Right Hip-Right Knee
            (2, 6),  # Right Hip - Pelvis
            (3, 6),  # Left Hip - Pelvis
            (3, 4),  # Left Hip - Left Knee
            (4, 5),  # Left Knee - Left Ankle
            (6, 7),  # Pelvis - Thorax
            (7, 8),  # Thorax - Upper Neck
            (8, 9),  # Upper Neck - Head Top
            (7, 12),  # Thorax - Right Shoulder
            (12, 11),  # Right Shoulder - Right Elbow
            (11, 10),  # Right Elbow - Right Wrist
            (7, 13),  # Thorax - Left Shoulder
            (13, 14),  # Left Shoulder - Left Elbow
            (14, 15)  # Left Elbow - Left Wrist
        ]
    elif data_type == 'coco' :
        edge_key = [
            (15, 13),  # left_ankle -> left_knee
            (13, 11),  # left_knee -> left_hip
            (16, 14),  # right_ankle -> right_knee
            (14, 12),  # right_knee -> right_hip
            (11, 12),  # left_hip -> right_hip
            (5, 11),   # left_shoulder -> left_hip
            (6, 12),   # right_shoulder -> right_hip
            (5, 6),    # left_shoulder -> right_shoulder
            (5, 7),    # left_shoulder -> left_elbow
            (6, 8),    # right_shoulder -> right_elbow
            (7, 9),    # left_elbow -> left_wrist
            (8, 10),   # right_elbow -> right_wrist
            (1, 2),    # left_eye -> right_eye
            (0, 1),    # nose -> left_eye
            (0, 2),    # nose -> right_eye
            (1, 3),    # left_eye -> left_ear
            (2, 4),    # right_eye -> right_ear
            (3, 5),    # left_ear -> left_shoulder
            (4, 6) ]   # right_ear -> right_shoulder
    elif data_type == 'coco_wholebody' :
        edge_key = [
            (15, 13),
            (13, 11),
            (16, 14),
            (14, 12),
            (11, 12),
            (5, 11),
            (6, 12),
            (5, 6),
            (5, 7),
            (6, 8),
            (7, 9),
            (8, 10),
            (1, 2),
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (3, 5),
            (4, 6),
            (15, 17),
            (15, 18),
            (15, 19),
            (16, 20),
            (16, 21),
            (16, 22),
            (91, 92),
            (92, 93),
            (93, 94),
            (94, 95),
            (91, 96),
            (96, 97),
            (97, 98),
            (98, 99),
            (91, 100),
            (100, 101),
            (101, 102),
            (102, 103),
            (91, 104),
            (104, 105),
            (105, 106),
            (106, 107),
            (91, 108),
            (108, 109),
            (109, 110),
            (110, 111),
            (112, 113),
            (113, 114),
            (114, 115),
            (115, 116),
            (112, 117),
            (117, 118),
            (118, 119),
            (119, 120),
            (112, 121),
            (121, 122),
            (122, 123),
            (123, 124),
            (112, 125),
            (125, 126),
            (126, 127),
            (127, 128),
            (112, 129),
            (129, 130),
            (130, 131),
            (131, 132)
        ]

    elif data_type == 'ap10k' :
        edge_key =  [
    (0, 1),    # L_Eye - R_Eye
    (0, 2),    # L_Eye - Nose
    (1, 2),    # R_Eye - Nose
    (2, 3),    # Nose - Neck
    (3, 4),    # Neck - Root of tail
    (3, 5),    # Neck - L_Shoulder
    (5, 6),    # L_Shoulder - L_Elbow
    (6, 7),    # L_Elbow - L_F_Paw
    (3, 8),    # Neck - R_Shoulder
    (8, 9),    # R_Shoulder - R_Elbow
    (9, 10),   # R_Elbow - R_F_Paw
    (4, 11),   # Root of tail - L_Hip
    (11, 12),  # L_Hip - L_Knee
    (12, 13),  # L_Knee - L_B_Paw
    (4, 14),   # Root of tail - R_Hip
    (14, 15),  # R_Hip - R_Knee
    (15, 16),  # R_Knee - R_B_Paw
                ]
    elif data_type == 'animalpose' :
        edge_key = [
    (0, 1),    # L_Eye - R_Eye
    (0, 2),    # L_Eye - L_EarBase
    (1, 3),    # R_Eye - R_EarBase
    (0, 4),    # L_Eye - Nose
    (1, 4),    # R_Eye - Nose
    (4, 5),    # Nose - Throat
    (5, 7),    # Throat - Withers
    (6, 7),    # TailBase - Withers
    (5, 8),    # Throat - L_F_Elbow
    (8, 12),   # L_F_Elbow - L_F_Knee
    (12, 16),  # L_F_Knee - L_F_Paw
    (5, 9),    # Throat - R_F_Elbow
    (9, 13),   # R_F_Elbow - R_F_Knee
    (13, 17),  # R_F_Knee - R_F_Paw
    (6, 10),   # TailBase - L_B_Elbow
    (10, 14),  # L_B_Elbow - L_B_Knee
    (14, 18),  # L_B_Knee - L_B_Paw
    (6, 11),   # TailBase - R_B_Elbow
    (11, 15),  # R_B_Elbow - R_B_Knee
    (15, 19)   # R_B_Knee - R_B_Paw
            ]

    edge_key = torch.tensor(edge_key, dtype=torch.long).T
    edge_key = edge_key.to(device)
    batch_size, num_keypoints, _  = inputs.shape
    data_list = []

    for b in range(batch_size):
        x = inputs[b].to(device)
        data = Data(x=x, edge_index=edge_key)
        data_list.append(data)

    batch = Batch.from_data_list(data_list)

    return batch

def bulid_keypoints_adj_matrix_reverse(inputs: torch.Tensor, data_type, device):

    if data_type == 'taiji' :
        #taiji
        edge_key = [
            [2, 0], [22, 0], [2, 1], [22, 21], [23, 1], [23, 21],
            [3, 2], [3, 22], [4, 2], [4, 22], [5, 4], [24, 4],
            [6, 5], [25, 24], [7, 5], [26, 24], [7, 6], [26, 25],
            [8, 6], [27, 25], [12, 8], [28, 27], [13, 12], [31, 28],
            [14, 13], [15, 13], [32, 31], [33, 31], [9, 7], [39, 26],
            [39, 9], [10, 9], [40, 39], [40, 10], [11, 10], [40, 29],
            [30, 11], [30, 29], [30, 4], [16, 11], [34, 29], [17, 16],
            [35, 34], [18, 17], [19, 17], [20, 17], [36, 35], [37, 35],
            [38, 35]
        ]
    elif data_type == 'mpii' :
        edge_key = [
            (0, 1),  # Right Ankle - Right Knee
            (1, 2),  # Right Knee - Right Hip
            (6, 2),  # Pelvis - Right Hip
            (6, 3),  # Pelvis - Left Hip
            (4, 3),  # Left Knee - Left Hip
            (5, 4),  # Left Ankle - Left Knee
            (7, 6),  # Thorax - Pelvis
            (8, 7),  # Upper Neck - Thorax
            (9, 8),  # Head Top - Upper Neck
            (12, 7),  # Right Shoulder - Thorax
            (11, 12),  # Right Elbow - Right Shoulder
            (10, 11),  # Right Wrist - Right Elbow
            (13, 7),  # Left Shoulder - Thorax
            (14, 13),  # Left Elbow - Left Shoulder
            (15, 14)  # Left Wrist - Left Elbow
        ]

    elif data_type == 'coco' :
        edge_key = [
            (13, 15),  # left_knee -> left_ankle
            (11, 13),  # left_hip -> left_knee
            (14, 16),  # right_knee -> right_ankle
            (12, 14),  # right_hip -> right_knee
            (12, 11),  # right_hip -> left_hip
            (11, 5),  # left_hip -> left_shoulder
            (12, 6),  # right_hip -> right_shoulder
            (6, 5), # right_shoulder -> left_shoulder
            (7, 5),  # left_elbow -> left_shoulder
            (8, 6),  # right_elbow -> right_shoulder
            (9, 7),  # left_wrist -> left_elbow
            (10, 8),  # right_wrist -> right_elbow
            (2, 1),  # right_eye -> left_eye
            (1, 0),  # left_eye -> nose
            (2, 0),  # right_eye -> nose
            (3, 1),  # left_ear -> left_eye
            (4, 2),  # right_ear -> right_eye
            (5, 3),  # left_shoulder -> left_ear
            (6, 4),  # right_shoulder -> right_ear
        ]
    elif data_type == 'coco_wholebody' :
        edge_key = [
            (13, 15),
            (11, 13),
            (14, 16),
            (12, 14),
            (12, 11),
            (11, 5),
            (12, 6),
            (6, 5),
            (7, 5),
            (8, 6),
            (9, 7),
            (10, 8),
            (2, 1),
            (1, 0),
            (2, 0),
            (3, 1),
            (4, 2),
            (5, 3),
            (6, 4),
            (17, 15),
            (18, 15),
            (19, 15),
            (20, 16),
            (21, 16),
            (22, 16),
            (92, 91),
            (93, 92),
            (94, 93),
            (95, 94),
            (96, 91),
            (97, 96),
            (98, 97),
            (99, 98),
            (100, 91),
            (101, 100),
            (102, 101),
            (103, 102),
            (104, 91),
            (105, 104),
            (106, 105),
            (107, 106),
            (108, 91),
            (109, 108),
            (110, 109),
            (111, 110),
            (113, 112),
            (114, 113),
            (115, 114),
            (116, 115),
            (117, 112),
            (118, 117),
            (119, 118),
            (120, 119),
            (121, 112),
            (122, 121),
            (123, 122),
            (124, 123),
            (125, 112),
            (126, 125),
            (127, 126),
            (128, 127),
            (129, 112),
            (130, 129),
            (131, 130),
            (132, 131)
        ]

    elif data_type == 'ap10k' :
        edge_key =  [
    (1, 0),    # R_Eye - L_Eye
    (2, 0),    # Nose - L_Eye
    (2, 1),    # Nose - R_Eye
    (3, 2),    # Neck - Nose
    (4, 3),    # Root of tail - Neck
    (5, 3),    # L_Shoulder - Neck
    (6, 5),    # L_Elbow - L_Shoulder
    (7, 6),    # L_F_Paw - L_Elbow
    (8, 3),    # R_Shoulder - Neck
    (9, 8),    # R_Elbow - R_Shoulder
    (10, 9),   # R_F_Paw - R_Elbow
    (11, 4),   # L_Hip - Root of tail
    (12, 11),  # L_Knee - L_Hip
    (13, 12),  # L_B_Paw - L_Knee
    (14, 4),   # R_Hip - Root of tail
    (15, 14),  # R_Knee - R_Hip
    (16, 15),  # R_B_Paw - R_Knee
                ]
    elif data_type == 'animalpose':
        edge_key = [
            (1, 0),  # R_Eye - L_Eye
            (2, 0),  # L_EarBase - L_Eye
            (3, 1),  # R_EarBase - R_Eye
            (4, 0),  # Nose - L_Eye
            (4, 1),  # Nose - R_Eye
            (5, 4),  # Throat - Nose
            (7, 5),  # Withers - Throat
            (7, 6),  # Withers - TailBase
            (8, 5),  # L_F_Elbow - Throat
            (12, 8),  # L_F_Knee - L_F_Elbow
            (16, 12),  # L_F_Paw - L_F_Knee
            (9, 5),  # R_F_Elbow - Throat
            (13, 9),  # R_F_Knee - R_F_Elbow
            (17, 13),  # R_F_Paw - R_F_Knee
            (10, 6),  # L_B_Elbow - TailBase
            (14, 10),  # L_B_Knee - L_B_Elbow
            (18, 14),  # L_B_Paw - L_B_Knee
            (11, 6),  # R_B_Elbow - TailBase
            (15, 11),  # R_B_Knee - R_B_Elbow
            (19, 15)  # R_B_Paw - R_B_Knee
        ]
    edge_key = torch.tensor(edge_key, dtype=torch.long).T
    edge_key = edge_key.to(device)
    batch_size, num_keypoints, _ = inputs.shape
    data_list = []

    for b in range(batch_size):
        x = inputs[b].to(device)
        data = Data(x=x, edge_index=edge_key)
        data_list.append(data)

    batch = Batch.from_data_list(data_list)

    return batch


