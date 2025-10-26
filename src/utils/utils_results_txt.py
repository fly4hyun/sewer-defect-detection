import os
import torch
import yaml

def save_results_to_files(cm, class_names, overall_metrics, class_metrics, save_folder, defect_map):
    """
    혼동 행렬과 메트릭을 txt와 yaml 파일로 저장합니다.
    :param cm: Confusion Matrix (Tensor 또는 Numpy array)
    :param class_names: 클래스 이름 딕셔너리
    :param overall_metrics: 전체 메트릭 딕셔너리
    :param class_metrics: 클래스별 메트릭 딕셔너리
    :param save_folder: 저장할 폴더명
    """
    # 딕셔너리에서 값만 추출해서 리스트로 변환
    class_names_list = [class_names[key] for key in sorted(class_names.keys())]
    class_names_list += ['backg']
    
    # Tensor일 경우 numpy로 변환
    cm_array = cm.cpu().numpy() if isinstance(cm, torch.Tensor) else cm  
    
    # 출력 내용을 저장할 리스트 초기화
    output_lines = []
    
    # Confusion Matrix 출력
    output_lines.append("Confusion Matrix:")
    output_lines.append(" " * 12 + "Predicted")
    output_lines.append(" " * 6 + " " + "  ".join(f"{name[:5]:>7}" for name in class_names_list))
    output_lines.append("Actual")
    
    for i, row in enumerate(cm_array):
        row_values = "  ".join(f"{val:7d}" for val in row)
        output_lines.append(f"{class_names_list[i][:5]:<7} | {row_values}")
    
    output_lines.append("")
    
    # Per-Class Metrics 출력
    output_lines.append("=== Per-Class Metrics ===")
    for i, (p, r, f1) in enumerate(zip(class_metrics['precision'], class_metrics['recall'], class_metrics['f1_score'])):
        output_lines.append(f"Class {defect_map.get_defect_name(int(i))}:")
        output_lines.append(f"  Precision: {p:.4f}")
        output_lines.append(f"  Recall:    {r:.4f}")
        output_lines.append(f"  F1 Score:  {f1:.4f}")
        output_lines.append("")
    
    # Overall Metrics 출력
    output_lines.append("=== Overall Metrics ===")
    output_lines.append(f"Overall Precision: {overall_metrics['precision']:.4f}")
    output_lines.append(f"Overall Recall:    {overall_metrics['recall']:.4f}")
    output_lines.append(f"Overall F1 Score:  {overall_metrics['f1_score']:.4f}")
    
    # txt 파일로 저장
    with open(os.path.join(save_folder, 'results.txt'), 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + '\n')
    
    # 데이터 yaml로 저장
    data_to_save = {
        'confusion_matrix': cm_array.tolist(),
        'class_names': class_names_list,
        'overall_metrics': overall_metrics,
        'class_metrics': class_metrics
    }
    with open(os.path.join(save_folder, 'results.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(data_to_save, f, allow_unicode=True)
