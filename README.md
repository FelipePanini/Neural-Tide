# Neural Tide

Arte visual interativa feita só com código: a webcam detecta suas mãos e efeitos
com tema de tecnologia/IA reagem a elas em tempo real, com movimento fluido,
como ondas do mar. A mão perturba e influencia, mas não controla diretamente.

> Projeto em construção, por fases. Este README cobre as **Fases 1 a 3**
> (fundação, modos 1 a 3, gestos e transições). A versão final, em português e
> inglês, com GIFs de cada modo, chega na Fase 5.

## Stack e versões

| Pacote | Versão | Função |
|---|---|---|
| **Python** | **3.12** | versão oficial do projeto |
| mediapipe | 1.2.0 | HandLandmarker (API Tasks, modo VIDEO) |
| opencv-contrib-python | 4.12.0.88 | só captura da webcam |
| numpy | 2.2.6 | simulação vetorizada |
| moderngl | 5.13.0 | renderização na GPU (OpenGL 3.3) |
| pygame-ce | 2.5.8 | janela, teclado e mouse |

As mesmas versões valem para Windows e macOS.

- **Python 3.12**: o mediapipe 1.2.0 suporta de 3.9 a 3.14, e todos os pacotes
  acima têm wheels para Windows e Mac na 3.12 (a 3.13 também deve funcionar).
- **OpenCV 4.12**: a partir da 4.13, o wheel de macOS traz uma cópia própria do
  SDL2 que conflita com a do pygame. O OpenCV vem como `opencv-contrib-python`
  porque é o pacote que o mediapipe já instala; adicionar `opencv-python` criaria
  dois `cv2` em conflito.
- **numpy 2.2**: o OpenCV 4.12 exige numpy abaixo de 2.3.

### Sistemas suportados

- **Windows 10/11** (64 bits)
- **macOS em Apple Silicon** (M1 ou mais novo). O mediapipe 1.x só publica wheels
  de macOS para arm64, então Macs com Intel não são suportados.

## Instalação

### Windows

```bat
git clone https://github.com/FelipePanini/Neural-Tide.git
cd Neural-Tide
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
python tools/download_model.py
```

### macOS

Instale o Python 3.12 (pelo [python.org](https://www.python.org/downloads/),
com `brew install python@3.12` ou com `uv python install 3.12`) e depois:

```bash
git clone https://github.com/FelipePanini/Neural-Tide.git
cd Neural-Tide
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python tools/download_model.py
```

Na primeira vez que você rodar com a webcam, o macOS pede permissão de câmera
para o app que abriu o Python (Terminal, iTerm, VS Code…). Se você negar, libere
depois em **Ajustes do Sistema › Privacidade e Segurança › Câmera** e reabra o
terminal. Se um iPhone por perto virar a câmera 0 (Câmera de Continuidade), use
`--camera 1`.

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
| 1 a 8 | troca de modo, com transição suave (por enquanto existem os modos 1 a 3) |
| P | próxima paleta de cores (neon, matrix, âmbar, gelo) |
| M | foco nas mãos ↔ tela cheia |
| F1 ou D | liga/desliga o overlay de debug (no Mac, F1 costuma exigir Fn) |
| C | fundo: câmera escurecida ou preto |
| ESC | sair |

O overlay de debug começa desligado. Ele mostra FPS, landmarks, o gesto
reconhecido em cada mão, intensidade, paleta e o tempo da simulação. Se o FPS
ficar abaixo de 30, diminua `DataTideConfig.particle_count` no `config.py`.

### Gestos

| Gesto | Ação |
|---|---|
| 1 a 5 dedos levantados, mão parada por 0,5 s | ir para o modo 1 a 5 |
| Swipe lateral com a mão aberta | próximo modo (para a esquerda) / anterior (para a direita) |
| Pinça (polegar + indicador) | interação do modo atual |
| Punho fechado | explosão |
| Chifrinho (indicador + mindinho) | próxima paleta de cores |
| Distância entre as duas mãos | intensidade |
| Palma aberta parada por 1 s | congelar / descongelar |

Enquanto você segura um gesto, um anel de progresso animado aparece na ponta do
dedo, com o que vai acontecer escrito ao lado. Quando o gesto dispara, o anel se
desfaz num flash e um aviso aparece no topo da tela.

- **Palma aberta**: 5 dedos significam tanto "modo 5" quanto "congelar", então o
  tempo de segurar decide. Se você soltar entre 0,5 s e 1 s, vai para o modo 5; se
  segurar até 1 s, congela. O anel mostra um marcador na metade. Enquanto o modo 5
  não existir, a palma só congela.
- **Chifrinho × 2 dedos**: indicador + mindinho é sempre chifrinho, com ou sem o
  polegar (🤘 e 🤟). "2 dedos" é qualquer outro par.
- **Pinça × punho**: a pinça exige que o indicador esteja esticado para a frente.
  No punho ele fica curvado, então o polegar encostando nele não conta como pinça.
  Durante a pinça a contagem de dedos é ignorada; assim o "OK" 👌 não vira
  "3 dedos".
- **Congelar** não para de uma vez: o tempo da simulação desacelera até zero (e
  acelera de volta).

### Modo mock

| Entrada | Mão simulada |
|---|---|
| mover o mouse | move a mão (o ponteiro fica no centro da palma) |
| roda do mouse, rolagem do trackpad ou setas ↑/↓ | quantidade de dedos levantados (0 a 5) |
| botão esquerdo / clique | pinça |
| botão direito ou F (segurado) | punho fechado |
| G (segurado) | chifrinho |
| H | liga/desliga uma segunda mão espelhada |
| tirar o mouse da janela | a mão sai do quadro |

## Como funciona

```
Camera (thread) ──► HandTracker (thread) ──► latest() ──► loop de render (60+ fps)
  captura e espelha    MediaPipe VIDEO         TrackingFrame    HandStateSmoother (easing)
  a imagem             slots + One Euro                         camadas: fundo → efeitos → debug
```

- **Captura em thread**: o render sempre usa o frame mais novo, sem esperar a câmera.
  No macOS, a primeira abertura acontece na thread principal, porque só assim o
  sistema mostra o pedido de permissão; depois a captura segue na thread.
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

### Modo 1: Maré de dados

24 mil partículas levadas por uma corrente. A mão agita a água, mas não arrasta
as partículas diretamente.

- **Curl noise**: um ruído simplex 3D (x, y, tempo), vetorizado em NumPy, gera um
  potencial numa grade grossa. A velocidade é o *rotacional* desse potencial,
  então o fluxo não tem fontes nem ralos: as partículas formam correntes e
  redemoinhos, como água, sem se amontoar. Cada partícula lê a grade por
  interpolação bilinear, e por isso o custo do ruído não cresce com o número
  de partículas.
- **Mola amortecida na velocidade**: cada partícula relaxa em direção à
  velocidade do fluxo local (constante de tempo que varia de partícula para
  partícula). Nada muda de velocidade de uma vez, e há inércia: um redemoinho
  rápido joga partículas para fora e abre um "olho", como num furacão.
- **A mão** é um vórtice de Lamb–Oseen (giro suave, sem singularidade no centro)
  mais uma corrente na direção do movimento dela. A mão direita gira no sentido
  anti-horário e a esquerda no horário. Mover a mão rápido deixa uma esteira.
- **Pinça**: o vórtice aperta e vira um ralo. As partículas se juntam num anel
  luminoso e somem perto do centro.
- **Ocioso**: sem mãos, um vórtice fantasma bem fraco passeia num "oito", e a maré
  respira devagar e muda de direção, então a tela nunca congela.
- **Ciclo de vida**: partículas nascem e morrem com fade, e quem sai da tela
  renasce em outro ponto, invisível no começo. Nenhuma posição aparece
  "teletransportada".
- **Punho**: um impulso de velocidade para fora (as posições continuam
  contínuas) e um flash de luz.

### Modo 2: Rede neural viva

Cerca de 170 nós flutuam numa corrente lenta e se ligam quando ficam próximos
(plexus). Uma repulsão suave mantém o espaçamento.

- **Pulso**: a pinça dispara o nó mais próximo. O pulso pula de nó em nó pelas
  ligações, com velocidade constante e perdendo energia a cada salto, então se
  espalha pela rede como uma onda e se apaga. Cada nó tem um período refratário,
  como um neurônio, e por isso a onda não volta para trás.
- **A mão** abre espaço na rede e a arrasta quando se move. As pontas dos dedos
  se conectam aos nós próximos, e encostar num nó dispara um pulso pequeno.
- **Ocioso**: pulsos espontâneos de vez em quando mantêm a rede viva.

### Modo 3: Grade holográfica

Uma malha em perspectiva que vai até o horizonte e é uma equação de onda 2D
de verdade (`h_tt = c²∇²h − amortecimento·h_t`), com subpassos para respeitar a
condição de estabilidade (CFL). Uma camada de amortecimento nas bordas absorve
as ondas que saem, em vez de deixá-las refletir.

- **Dedos**: cada dedo levantado empurra a superfície no ponto logo abaixo dele (o
  raio da câmera que passa pela ponta do dedo encontra o plano ali). Dedo parado
  faz uma covinha; dedo em movimento deixa uma esteira.
- **Pinça** solta uma onda e o **punho** solta uma onda grande.
- **Ocioso**: gotas de chuva aleatórias e um balanço suave de oceano.
- **Visual**: as linhas são calculadas no fragment shader a partir das coordenadas
  da grade, com `fwidth`. Assim elas têm espessura constante em pixels a qualquer
  distância e somem antes de virar moiré no horizonte.

### Foco nas mãos

Por padrão, os efeitos aparecem só em volta das mãos, para a tela não ficar
poluída. Cada mão abre um "holofote" de borda suave, do tamanho da mão aberta
na tela (`FocusConfig.hand_scale`), que segue a mão com a mesma suavidade do
rastreio. Longe das mãos sobra só um resquício de luz (`ambient`). Cada modo
ajusta o tamanho do holofote; a grade, por exemplo, usa uma janela mais larga.
Sem mãos, um holofote menor e mais fraco passeia devagar, então o estado ocioso
continua vivo. A simulação segue acontecendo na tela toda: a máscara só decide
o que aparece. A tecla **M** alterna, com fade, entre o foco e a tela cheia.

### Transições

- **Crossfade**: os modos continuam vivos e só a opacidade muda, com easing. O
  modo que sai e o que entra se misturam no mesmo buffer de rastros.
- **Ondulação**: cada troca de modo dispara uma onda que atravessa a camada de
  efeitos a partir da mão que fez o gesto (ou do centro, se veio do teclado).
- **Rastros por modo**: a Maré usa rastros longos, a rede usa médios e a grade
  quase nenhum. O tempo dos rastros também passa por crossfade.

### Pós-processamento

- **Rastros**: a camada de efeitos fica num buffer HDR (float16) que nunca é
  limpo. A cada frame, o frame anterior é multiplicado por `exp(-dt/τ)` (decaimento
  independente do FPS) e se espalha um pouco para os vizinhos.
- **Partículas como riscos**: cada partícula é um quad instanciado e esticado na
  direção da velocidade (motion blur), desenhado com mistura aditiva.
- **Bloom** em cadeia de mipmaps com filtro duplo (dual Kawase) e limiar suave.
- **Tone mapping que preserva a cor**: o canal mais forte é comprimido e o matiz é
  mantido, então o neon denso continua neon em vez de virar branco. No fim entra
  um dither leve contra banding.
- **Brilho independente do FPS**: num buffer com decaimento, a luz se acumula até
  `1/(1 − decaimento)` vezes, e esse valor depende do FPS. Um ganho por frame
  compensa isso, então a imagem fica igual a 30 ou a 144 FPS.

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
  gestures.py        máquina de estados dos gestos
sim/
  noise.py           ruído simplex 3D vetorizado
  flow_field.py      campo de curl noise em grade, amostragem bilinear
  forces.py          vórtice, corrente e sucção (vetorizados)
  waves.py           equação de onda 2D
effects/
  base.py            classe Effect: enter(), exit(), update(dt, hands), draw()
  __init__.py        registro de modos (EFFECTS)
  manager.py         crossfade, congelar, intensidade
  data_tide.py       modo 1: Maré de dados
  neural_net.py      modo 2: Rede neural viva
  holo_grid.py       modo 3: Grade holográfica
render/
  renderer.py        ordem do frame, fundo da câmera
  postfx.py          rastros, bloom, tone mapping e ondulação de transição
  gesture_ui.py      anel de progresso, avisos e selos
  focus.py           holofotes que limitam o efeito às mãos
  projection.py      perspectiva e raio da tela até o plano
  glow.py            manchas de luz aditivas
  palette.py         paleta de cores com easing
  overlay.py         overlay de debug (F1)
  primitives.py      pontos suaves e linhas com anti-aliasing
  text.py            texto via pygame.font em texturas
  shaders/*.glsl
tools/download_model.py
tests/
```

## Roadmap

- [x] **Fase 1**: fundação (captura, tracking, One Euro, janela ModernGL, debug, mock)
- [x] **Fase 2**: modo 1 (Maré de dados) com curl noise, rastros e bloom
- [x] **Fase 3**: máquina de gestos, anel de progresso, transições suaves, modos 2 e 3
  (as 4 paletas vieram junto, porque o chifrinho precisava delas)
- [ ] Fase 4: modos 4 a 8
- [ ] Fase 5: gravação MP4 e README final

## Licença

MIT. Veja [LICENSE](LICENSE).
