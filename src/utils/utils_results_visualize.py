import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os

plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False  # 유니코드에서 음수 부호 설정

# 클래스 이름과 색상 정의
class_names_dict = {
    0: '균열', 1: '연결관', 2: '이음부', 3: '토사퇴적', 4: '파손',
    5: '표면손상', 6: '라이닝결함', 7: '좌굴', 8: '변형', 9: '영구장애물',
    10: '천공', 11: '침하', 12: '폐유부착', 13: '임시장애물', 14: '뿌리침입'
}

class_colors_bgr = {
    0: (180, 180, 180),
    1: (255, 191, 0),
    2: (0, 165, 255),
    3: (50, 205, 50),
    4: (0, 0, 255),
    5: (0, 215, 255),
    6: (221, 160, 221),
    7: (255, 144, 30),
    8: (0, 140, 255),
    9: (128, 0, 128),
    10: (147, 20, 255),
    11: (19, 69, 139),
    12: (105, 105, 105),
    13: (180, 105, 255),
    14: (127, 255, 0),
}

# BGR에서 RGB로 변환
class_colors = {key: (value[2]/255, value[1]/255, value[0]/255) for key, value in class_colors_bgr.items()}

def plot_confusion_matrix(cm, class_names_dict, save_path=None, normalize=False, figsize=(10, 8)):
    # 클래스 이름 리스트 생성
    class_names = [class_names_dict[key] for key in sorted(class_names_dict.keys())]

    # Tensor를 numpy 배열로 변환
    cm = cm.cpu().numpy() if isinstance(cm, torch.Tensor) else cm

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=figsize)

    # 클래스별 색상 매핑
    # 클래스 인덱스가 class_colors에 있는 경우에만 색상을 가져옵니다.
    colors = []
    for i in range(len(class_names)):
        if i in class_colors:
            colors.append(class_colors[i])
        else:
            colors.append((0.5, 0.5, 0.5))  # 기본 색상(회색) 사용

    # 히트맵 그리기
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.ylabel('실제 값')
    plt.xlabel('예측 값')
    plt.title('혼동 행렬')
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'results_cm.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_metrics(class_metrics, class_names_dict, save_path=None, figsize=(12, 6)):
    class_names = [class_names_dict[key] for key in sorted(class_names_dict.keys())]
    num_classes = len(class_names)
    x = np.arange(num_classes)
    width = 0.25

    metrics = ['precision', 'recall', 'f1_score']
    colors = [class_colors[i] for i in range(num_classes)]

    plt.figure(figsize=figsize)

    for idx, metric in enumerate(metrics):
        plt.bar(x + idx * width, class_metrics[metric], width, label=metric.upper(), color=colors, alpha=0.7)

    plt.xticks(x + width, class_names, rotation=45)
    plt.ylim(0, 1)
    plt.ylabel('점수')
    plt.title('클래스별 성능 지표')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'results_class.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_overall_metrics(overall_metrics, save_path=None):
    metrics = ['precision', 'recall', 'f1_score']
    scores = [overall_metrics[metric] for metric in metrics]
    colors = ['skyblue', 'lightgreen', 'salmon']

    plt.figure(figsize=(6, 4))
    plt.bar(metrics, scores, color=colors)
    plt.ylim(0, 1)
    plt.ylabel('점수')
    plt.title('전체 성능 지표')
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'results_overall.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics(class_metrics, overall_metrics, class_names_dict, save_path=None, figsize=(12, 6)):
    class_names = [class_names_dict[key] for key in sorted(class_names_dict.keys())]
    num_classes = len(class_metrics['precision'])

    # 클래스 이름과 메트릭 길이 맞추기
    class_names = class_names[:num_classes]

    # 'Overall' 추가
    class_names.append('전체')

    metrics = ['precision', 'recall', 'f1_score']
    x = np.arange(num_classes + 1)
    width = 0.25

    plt.figure(figsize=figsize)

    for idx, metric in enumerate(metrics):
        # 클래스별 메트릭에 전체 메트릭 추가
        metric_values = class_metrics[metric] + [overall_metrics[metric]]
        colors = [class_colors[i] if i < num_classes else 'gray' for i in range(num_classes + 1)]

        plt.bar(x + idx * width, metric_values, width, label=metric.upper(), color=colors, alpha=0.7)

        # 막대 위에 값 표시
        for xi, yi in zip(x + idx * width, metric_values):
            plt.text(xi, yi + 0.01, f'{yi:.2f}', ha='center', va='bottom', fontsize=9)

    y_max = max([max(class_metrics[metric] + [overall_metrics[metric]]) for metric in metrics]) + 0.1
    y_max = 1.05 if y_max > 1 else y_max

    plt.xticks(x + width, class_names, rotation=45)
    plt.ylim(0, y_max)
    plt.ylabel('점수')
    plt.title('클래스별 및 전체 성능 지표')
    plt.legend(loc='lower right')
    plt.yticks(np.arange(0, y_max + 0.05, 0.1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'results_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()