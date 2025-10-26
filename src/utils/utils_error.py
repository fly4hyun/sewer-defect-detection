from utils.utils_api import create_output, send_to_api

def handle_error(error_type, data, video_url, key):
    if error_type == 'UnboundLocalError':
        results_pipe_info = {
            'pipe_start': None,
            'pipe_end': None,
            'pipe_number': None,
            'date': None
        }
        output = create_output(results_pipe_info, data, video_url, key, msg='UnboundLocalError')
        send_to_api(output)