# Neural Tide

Arte visual interativa feita só com código: a webcam detecta suas mãos e efeitos
com tema de tecnologia/IA reagem a elas em tempo real, com movimento fluido,
como ondas do mar. A mão perturba e influencia, mas não controla diretamente.

> Projeto em construção, por fases. Este README cobre a **Fase 1 (fundação)**.
> A versão final, em português e inglês, com GIFs de cada modo, chega na Fase 5.

## Stack e versões

| Pacote | Versão | Função |
|---|---|---|
| **Python** | **3.12** | versão oficial do projeto |
| mediapipe | 1.2.0 | HandLandmarker (API Tasks, modo VIDEO) |
| opencv-contrib-python | 4.14.0.94 | só captura da webcam |
| numpy | 2.5.4 | simulação vetorizada |
| moderngl | 5.13.0 | renderização na GPU (OpenGL 3.3) |
| pygame-ce | 2.5.8 | janela, teclado e mouse |

Por que Python 3.12: o mediapipe 1.2.0 suporta de 3.9 a 3.14, mas o numpy 2.5
exige 3.12 ou mais novo. A 3.12 tem wheels para Windows de todos os pacotes acima
(a 3.13 também deve funcionar). O OpenCV vem como `opencv-contrib-python` porque é
o pacote que o mediapipe já instala; adicionar `opencv-python` criaria dois `cv2`
em conflito.

## Instalação (Windows)

```bat
git clone https://github.com/FelipePanini/Neural-Tide.git
cd Neural-Tide
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
python tools/download_model.py
```

`tools/download_model.py` baixa o `hand_landmarker.task` para `models/`.
Com `--all`, baixa também o segmentador de pessoa, usado no modo 7.

## Como rodar

| Comando | O que faz |
|---|---|
| `python main.py` | usa a webcam |
| `python main.py --mock` | o mouse vira uma mão (não precisa de webcam) |
| `python main.py --camera 1 --size 1920x1080 --fullscreen` | outra câmera, tamanho e tela cheia |
| `pytest` | roda os testes |

### Teclas

| Tecla | Ação |
|---|---|
| F1 | liga/desliga o overlay de debug |
| C | fundo: câmera escurecida ou preto |
| ESC | sair |

### Modo mock

| Entrada | Mão simulada |
|---|---|
| mover o mouse | move a mão (o ponteiro fica no centro da palma) |
| roda do mouse | quantidade de dedos levantados (0 a 5) |
| botão esquerdo | pinça |
| botão direito (segurado) | punho fechado |
| H | liga/desliga uma segunda mão espelhada |
| tirar o mouse da janela | a mão sai do quadro |

## Como funciona

```
Camera (thread) ──► HandTracker (thread) ──► latest() ──► loop de render (60+ fps)
  captura e espelha    MediaPipe VIDEO         TrackingFrame    HandStateSmoother (easing)
  a imagem             slots + One Euro                         camadas: fundo → efeitos → debug
```

- **Captura em thread**: o render sempre usa o frame mais novo, sem esperar a câmera.
- **Tracking em thread**: o MediaPipe roda fora do loop de render, então uma
  inferência lenta não derruba o FPS da animação.
- **Filtro One Euro** em todos os landmarks (imagem e 3D). Ele suaviza bem a mão
  parada e responde rápido quando ela se move. Os parâmetros ficam em
  `config.py` (`TrackerConfig.image_filter` / `world_filter`).
- **Easing no render**: o tracker roda a ~30 Hz e o render a 60+ Hz, então cada
  frame aproxima suavemente a mão da última posição filtrada. Quando a mão aparece
  ou some, a presença entra e sai com fade.
- **Esquerda/direita**: a imagem é espelhada (visão de selfie). O MediaPipe 1.2
  rotula a mão como se a foto não fosse espelhada, então os rótulos são trocados
  (`TrackerConfig.model_expects_mirrored`).
- **Contagem de dedos** com landmarks 3D (em metros): soma dos ângulos das
  articulações de cada dedo. Assim a contagem não depende da distância nem da rotação
  da mão. O polegar também precisa apontar para fora da palma.

## Estrutura

```
main.py              argumentos de linha de comando
app.py               loop principal
config.py            todos os parâmetros ajustáveis
core/
  camera.py          captura da webcam em thread
  tracker.py         MediaPipe HandLandmarker + slots + One Euro
  filters.py         filtro One Euro vetorizado
  hand.py            landmarks, dedos, pinça, tipos de dados
  hand_state.py      mãos no espaço do mundo, com easing por frame
  input_mock.py      mão sintética (mouse e testes)
  viewport.py        mapeamento imagem ↔ mundo ↔ pixels
effects/
  base.py            classe Effect: enter(), exit(), update(dt, hands), draw()
  __init__.py        registro de modos (EFFECTS)
render/
  renderer.py        fundo da câmera, estado da GPU
  overlay.py         overlay de debug (F1)
  primitives.py      pontos suaves e linhas com anti-aliasing
  text.py            texto via pygame.font em texturas
  shaders/*.glsl
tools/download_model.py
tests/
```

## Roadmap

- [x] **Fase 1**: fundação (captura, tracking, One Euro, janela ModernGL, debug, mock)
- [ ] Fase 2: modo 1 (Maré de dados) com curl noise, rastros e bloom
- [ ] Fase 3: máquina de gestos, transições suaves, modos 2 e 3
- [ ] Fase 4: modos 4 a 8 e paletas
- [ ] Fase 5: gravação MP4 e README final

## Licença

MIT. Veja [LICENSE](LICENSE).
