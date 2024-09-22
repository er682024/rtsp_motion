"""
RTSP Motion Detection

Questo modulo fornisce funzionalità per la rilevazione di movimento utilizzando
stream video RTSP. Il modulo analizza i frame del video per rilevare variazioni 
e segnala eventi di movimento.
"""

import base64
import cv2
import datetime
import json
import numpy as np
import os
import pytz
import requests
import sys
import time
from flask import Flask, jsonify, render_template_string, request
from pythonping import ping


app_name = 'rtsp_motion'
config_file_path = 'config/config.json'

ultimo_movimento = ultimo_frame = fuso_orario = None
presente = flag_fuori_orario = False
ip_cam = ip_plug = rtsp_url = url_on = url_off = ''
soglia_diff = soglia_pixel = soglia_tempo = dalle_ore = alle_ore = tempo_passato = 0
movimenti = []
MAX_MOVIMENTI = 5

app = Flask(__name__)

def carica_configurazione():
    config_default = {
        "FUSO_ORARIO": "Europe/Rome",
        "IP_CAM": "192.168.1.100",
        "IP_PLUG": "192.168.1.101",
        "RTSP_URL": "rtsp://username:password@192.168.1.100:554/stream1",
        "URL_ON": "http://192.168.1.101/relay/0%3Fturn%3Don",
        "URL_OFF": "http://192.168.1.101/relay/0%3Fturn%3Doff",
        "SOGLIA_DIFF": 1113,
        "SOGLIA_PIXEL": 150,
        "SOGLIA_TEMPO": 300,
        "DALLE_ORE": 7,
        "ALLE_ORE": 21
    }
    try:
        if not os.path.exists(config_file_path) or os.path.getsize(config_file_path) == 0:
            # Crea un file con la configurazione di default se non esiste o è vuoto
            with open(config_file_path, 'w') as f:
                json.dump(config_default, f)
            return config_default
        else:
            with open(config_file_path, 'r') as f:
                return json.load(f)
    except json.JSONDecodeError:
        # Se il file esiste ma è corrotto, riscrivi con il default
        with open(config_file_path, 'w') as f:
            json.dump(config_default, f)
        return config_default


def ora_locale():
    global fuso_orario
    # print(f'FUSO_ORARIO:>{fuso_orario}<', flush=True)
    if fuso_orario is None:
        return datetime.datetime.now()
    adesso = datetime.datetime.now(
            pytz.utc
            ).astimezone(
                    pytz.timezone(
                        fuso_orario
                        )
                    )
    return adesso

start_time = ora_locale()

def debug(testo, altro=''):
    # return
    global start_time
    data_ora_corrente = ora_locale()
    data_ora_formattata = data_ora_corrente.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{start_time} {data_ora_formattata} - {testo} {altro}", flush=True)


def print_env():
    global fuso_orario, url_on, url_off, soglia_diff, soglia_pixel
    global soglia_tempo, dalle_ore, alle_ore
    debug('FUSO_ORARIO_:', fuso_orario)
    debug('IP_CAM______:', ip_cam)
    debug('IP_PLUG_____:', ip_plug)
    debug('URL_ON______:', url_on)
    debug('URL_OFF_____:', url_off)
    debug('SOGLIA_DIFF_:', soglia_diff)
    debug('SOGLIA_PIXEL:', soglia_pixel)
    debug('SOGLIA_TEMPO:', soglia_tempo)
    debug('DALLE_ORE___:', dalle_ore)
    debug('ALLE_ORE____:', alle_ore)


def imposta_configurazione():
    global config
    global fuso_orario, ip_cam, ip_plug, rtsp_url, url_on, url_off
    global soglia_diff, soglia_pixel, soglia_tempo, dalle_ore, alle_ore
    fuso_orario = config['FUSO_ORARIO']
    ip_cam = config['IP_CAM']
    ip_plug = config['IP_PLUG']
    rtsp_url = config['RTSP_URL']
    url_on = config['URL_ON']
    url_off = config['URL_OFF']
    soglia_diff = config['SOGLIA_DIFF']
    soglia_pixel = config['SOGLIA_PIXEL']
    soglia_tempo = config['SOGLIA_TEMPO']
    dalle_ore = config['DALLE_ORE']
    alle_ore = config['ALLE_ORE']
    print_env()


def segnala_movimento(num_diff_pixels, soglia_diff, threshold, frame):
    global movimenti
    global ultimo_movimento
    global MAX_MOVIMENTI
    # debug(
    #         f'Movimento - diff:{num_diff_pixels} - '
    #         f'soglia_diff:{soglia_diff} - '
    #         f'threshold:{threshold} - '
    #         f'ultimo_movimento={ultimo_movimento}'
    #         )
    if len(movimenti) >= MAX_MOVIMENTI:
        movimenti.pop(0)
    ultimo_movimento = ora_locale()
    movimenti.append((ultimo_movimento, frame))


def cattura_frame(frame_prec, threshold):
    global rtsp_url
    global ultimo_movimento
    global ultimo_frame
    # debug(f'cap_read - ultimo_movimento={ultimo_movimento}')
    cap = cv2.VideoCapture(rtsp_url)
    ret, frame = cap.read()
    if not ret:
        if frame_prec is None:
            return False, None
        return False, frame_prec
    if frame_prec is None:
        return False, frame
    diff = cv2.absdiff(frame_prec, frame)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, diff_thresh = cv2.threshold(
            diff_gray,
            threshold,
            255,
            cv2.THRESH_BINARY
            )
    num_diff_pixels = cv2.countNonZero(diff_thresh)
    cap.release()
    # cv2.destroyAllWindows()
    return num_diff_pixels, frame


def rileva_movimento(frame_prec, soglia_diff, threshold):
    global rtsp_url
    global ultimo_movimento
    global ultimo_frame
    num_diff_pixels, frame = cattura_frame(frame_prec, threshold)
    if num_diff_pixels > soglia_diff:
        segnala_movimento(num_diff_pixels, soglia_diff, threshold, frame)
        return True, frame
    return False, frame


def chiedi(url):
    if ping_ko(ip_plug):
        debug('KO IP_PLUG:', ip_plug)
        return False
    ok = False
    try:
        response = requests.get(url)
        if response.status_code == 200:
            debug("Request OK!")
            ok = True
        else:
            debug(f"Request KO: {response.status_code}")
    except requests.exceptions.RequestException as e:
        debug(f"Request KO: {e}")
    return ok


def ping_ko(ip):
    try:
        response = ping(ip, count=1, timeout=2)
        if not response.success():
            debug(f'PING RESPONSE:{response}')
            return True
    except Exception as e:
        debug('PING EXCEPTION:', f'{e}')
        return True
    return False


def html_init(title, header, page):
    global app_name
    out = (
            '<!DOCTYPE html>'
            '<html lang="en">'
            '<head>'
            '<meta charset="UTF-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
            f'<title>{app_name} {title}</title>'
            '<link rel="stylesheet" href="{{ url_for(\'static\', filename=\'styles.css\') }}">'
            '</head>'
            '<body>'
            '<header>'
            f'<h1>{app_name} {header}</h1>'
            '</header>'
            '<div class="container">'
            f'<p>{app_name} {page}</p>'
            )
    return out


def html_close():
    return '</div></body></html>'

@app.route('/config', methods=['GET', 'POST'])
def configurazione():
    if request.method == 'POST':
        # Aggiorna la configurazione con i dati inviati dal form
        config['FUSO_ORARIO'] = request.form['fuso_orario']
        config['IP_CAM'] = request.form['ip_cam']
        config['IP_PLUG'] = request.form['ip_plug']
        config['RTSP_URL'] = request.form['rtsp_url']
        config['URL_ON'] = request.form['url_on']
        config['URL_OFF'] = request.form['url_off']
        config['SOGLIA_DIFF'] = int(request.form['soglia_diff'])
        config['SOGLIA_PIXEL'] = int(request.form['soglia_pixel'])
        config['SOGLIA_TEMPO'] = int(request.form['soglia_tempo'])
        config['DALLE_ORE'] = int(request.form['dalle_ore'])
        config['ALLE_ORE'] = int(request.form['alle_ore'])

        # Salva la configurazione nel file JSON
        with open(config_file_path, 'w') as f:
            json.dump(config, f)
        imposta_configurazione()
        return jsonify({'message': 'Configurazione aggiornata!'})

    # Renderizza la pagina di configurazione
    return render_template_string(
    html_init(title='config_tit', header='config_head', page='config_page') + '''
    <a href="/visualizza_movimento">Visualizza Movimento</a>
    <hr>
    <form method="POST">
        <label>Fuso Orario:</label><br>
        <input type="text" name="fuso_orario" value="{{config['FUSO_ORARIO']}}"><hr>

        <label>IP Cam:</label><br>
        <input type="text" name="ip_cam" value="{{config['IP_CAM']}}"><hr>

        <label>IP Plug:</label><br>
        <input type="text" name="ip_plug" value="{{config['IP_PLUG']}}"><hr>

        <label>RTSP URL:</label><br>
        <input type="text" name="rtsp_url" value="{{config['RTSP_URL']}}"><hr>

        <label>URL On:</label><br>
        <input type="text" name="url_on" value="{{config['URL_ON']}}"><button type="button" onclick="window.open('{{config['URL_ON']}}', '_blank')">Test</button><br>

        <label>URL Off:</label><br>
        <input type="text" name="url_off" value="{{config['URL_OFF']}}"><button type="button" onclick="window.open('{{config['URL_OFF']}}', '_blank')">Test</button><hr>

        <label>Soglia Diff:</label><br>
        <input type="number" name="soglia_diff" value="{{config['SOGLIA_DIFF']}}"><br>

        <label>Soglia Pixel:</label><br>
        <input type="number" name="soglia_pixel" value="{{config['SOGLIA_PIXEL']}}"><br>

        <label>Soglia Tempo:</label><br>
        <input type="number" name="soglia_tempo" value="{{config['SOGLIA_TEMPO']}}"><hr>

        <label>Dalle Ore:</label><br>
        <input type="number" name="dalle_ore" value="{{config['DALLE_ORE']}}"><br>

        <label>Alle Ore:</label><br>
        <input type="number" name="alle_ore" value="{{config['ALLE_ORE']}}"><br>

        <input type="submit" value="Salva">
    </form>
    ''' + html_close()
    , config=config)


@app.route('/ultimo_movimento', methods=['GET'])
def get_ultimo_movimento():
    global ultimo_movimento
    if ultimo_movimento is None:
        return jsonify({"ultimo_movimento": None})
    return jsonify(
            {
                "ultimo_movimento": ultimo_movimento.strftime(
                    "%Y-%m-%d %H:%M:%S"
                    )
                }
            )


@app.route('/visualizza_movimento', methods=['GET'])
def visualizza_movimento():
    global movimenti, presente, tempo_passato
    html_content = html_init(title='Ultimi Movimenti', header='ultimi movimenti', page='Ultimi movimenti') + (
            '<form action="/config" method="get">'
            '<button type="submit">Vai alla pagina di configurazione</button>'
            '</form>'
            '<hr>'
            f'<h2>Stato Presenza: {"Presente" if presente else "Assente"} {"fuori orario" if flag_fuori_orario else ""} {int(tempo_passato)}</h2>'
            '<hr>'
            )
    if len(movimenti) == 0:
        html_content += "Nessun movimento rilevato."
    else:
        for idx, (timestamp, frame) in enumerate(reversed(movimenti)):
            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            html_content += (
            f"<h2>Movimento {idx + 1}</h2>"
            f'<p>Data e ora: {timestamp.strftime("%Y-%m-%d %H:%M:%S")}</p>'
            f'<img src="data:image/jpeg;base64,{frame_b64}" '
            f'alt="Frame Movimento {idx + 1}">'
            '<br><br>'
            )
    html_content += html_close()
    return render_template_string(html_content)


def fuori_orario():
    global dalle_ore, alle_ore
    adesso = ora_locale()
    if adesso.hour < dalle_ore:
        debug(f'ORE {adesso.hour} < {dalle_ore}')
        return True
    if adesso.hour > alle_ore:
        debug(f'ORE {adesso.hour} > {alle_ore}')
        return True
    return False


def rip(secs):
    debug(f'sleep {secs} secs...')
    time.sleep(60)


def cambio_stato(presente):
    global flag_fuori_orario
    flag_fuori_orario = fuori_orario()
    if flag_fuori_orario:
        rip(60)
        return presente
    if presente:
        debug("False->True")
        if chiedi(url_on):
            return presente
    debug("True->False")
    if chiedi(url_off):
        return presente
    return not presente


def main():
    global rtsp_url
    global ultimo_movimento
    global soglia_tempo
    global url_on
    global url_off
    global dalle_ore
    global alle_ore
    global presente
    global tempo_passato
    # debug(f'{app_name} {rtsp_url} {soglia_diff} {soglia_pixel}')
    # debug(f'URL_ON  : {url_on}')
    # debug(f'URL_OFF : {url_off}')
    frame_prec = None
    tempo_passato = soglia_tempo + 1
    presente = True
    ultimo_movimento = datetime.datetime(
            year=1900,
            month=12,
            day=1
            ).astimezone(
                    pytz.timezone(
                        fuso_orario
                        )
                    )
    while True:
        if presente and frame_prec is not None:
            time.sleep(soglia_tempo)
        else:
            time.sleep(1)
        if ping_ko(ip_cam):
            debug('KO IP_CAM:', ip_cam)
            continue
        presente_prec = presente
        esito, frame_prec = rileva_movimento(
                frame_prec,
                soglia_diff,
                soglia_pixel
                )
        # debug(f'ultimo_movimento:{ultimo_movimento}')
        if ultimo_movimento is None:
            continue
        tempo_passato = (ora_locale() - ultimo_movimento).total_seconds()
        if tempo_passato > soglia_tempo:
            if presente:
                presente = cambio_stato(False)
            continue
        if not presente:
            presente = cambio_stato(True)

#------------------------------------------------------------------------------

config = carica_configurazione()
imposta_configurazione()

if __name__ == "__main__":
    from threading import Thread
    Thread(target=main).start()
    app.run(host='0.0.0.0', port=5000, debug=False)
