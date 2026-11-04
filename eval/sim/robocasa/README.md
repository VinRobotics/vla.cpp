# RoboCasa365 evaluation with SmolVLA and GR00T N1.6

This guide sets up the RoboCasa simulator, starts a C++ inference server, and
runs evaluation episodes with SmolVLA or GR00T N1.6. Use two terminals: keep the
server running in one and launch the simulator client in the other.

All commands below run from the **vla.cpp repository root**.

- [Set up the environment](#1-set-up-the-environment)
- [Run SmolVLA](#2-run-smolvla)
- [Run GR00T N1.6](#3-run-gr00t-n16)
- [Read results](#4-read-results)
- [Run inference on another machine](#5-run-inference-on-another-machine)
- [Valid task names](#6-valid-task-names)
- [Troubleshooting](#7-troubleshooting)

## 1. Set up the environment

### Prerequisites

The simulator setup requires `git`, `uv`, `protoc`, and working EGL/OpenGL system
libraries. The inference server additionally needs CMake, a C++ compiler,
protobuf, ZeroMQ, and cppzmq development libraries. For Debian/Ubuntu, the system
packages include:

```bash
sudo apt-get install -y git cmake build-essential protobuf-compiler \
    libprotobuf-dev libzmq3-dev cppzmq-dev libegl1-mesa-dev libglu1-mesa
```

Install `uv` separately if it is not already on `PATH`. The setup script installs
Python dependencies locally; it does not install system packages.

### Install the simulator and assets

```bash
bash eval/sim/robocasa/setup_robocasa.sh
```

The script:

1. Creates or reuses a Python 3.11 environment at
   `eval/sim/robocasa/robocasa_uv/.venv`.
2. Checks out pinned RoboCasa 1.0.1 and robosuite 1.5.2 revisions and installs
   the simulator dependencies, including MuJoCo 3.3.1 and CPU PyTorch.
3. Downloads the `tex`, `tex_generative`, `fixtures_lw`, and `objs_lw` asset packs.
4. Runs a rendered reset/step check without a model or inference server.

Asset downloads can take substantial time and disk space. Completed packs are
reused when setup runs again; incomplete packs are retried. The environment uses
Lightwheel objects, matching the downloaded object packs. Existing private macro
files are preserved.

The shell client automatically selects this Python environment; activation is
optional. Keep it separate from LIBERO's environment, which uses MuJoCo 2.x.
LeRobot and the Isaac-GR00T inference package are not required by this client.

The pinned source revisions are:

| Component | Revision |
|---|---|
| RoboCasa 1.0.1 | `3443cdf3dd20f9ae8ed41adba442059b506891d0` |
| robosuite 1.5.2 | `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |

`ROBOCASA_REV` and `ROBOSUITE_REV` override these pins. Setup refuses to switch
a reused checkout's revision when it has tracked edits. Installed Python
versions are recorded in `eval/sim/robocasa/robocasa_uv/installed.txt`.

To install dependencies first and defer asset downloads and rendering:

```bash
bash eval/sim/robocasa/setup_robocasa.sh --skip-assets --skip-smoke

# Later, finish asset installation and run the check.
bash eval/sim/robocasa/setup_robocasa.sh
```

### Check rendering and assets

Run this independently after setup, with no server or checkpoint:

```bash
bash eval/run_robocasa_client.sh \
    --check-env \
    --task CloseFridge \
    --max-episode-steps 2
```

Expected output includes:

```text
[check-env] CloseFridge: reset/step/render OK
```

The default renderer is EGL and requires access to the graphics device and its
driver. Python policy preprocessing runs on CPU. For an installed OSMesa
software renderer, set `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa` when launching
the setup script or client.

### Build the inference server

If `build/vla-server` already exists, continue to the model instructions.
For a fresh CPU build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVLA_BUILD_SERVER=ON
cmake --build build --target vla-server -j"$(nproc)"
```

For GPU inference, use the CUDA or other backend build instructions in the
[repository README](../../../README.md#build-from-source), then use the resulting
`vla-server`. Set `SERVER_BIN=/absolute/path/to/vla-server` when using a binary
outside `build/`.

## 2. Run SmolVLA

The server loads `weights/smolvla_robocasa.gguf` by default. Setup installs the
simulator only; it does not download or convert this checkpoint.

### Download and convert the SmolVLA checkpoint

Download `lerobot/smolvla_robocasa` at the revision the client reads its
metadata from, then convert it to GGUF:

```bash
# Download the checkpoint from Hugging Face
huggingface-cli download lerobot/smolvla_robocasa \
    --revision f7af588d11e30b0069f200792fb285904b74c80b \
    --local-dir weights/smolvla_robocasa

# Convert to GGUF
python scripts/convert_smolvla_to_gguf.py \
    --ckpt weights/smolvla_robocasa \
    --out weights/smolvla_robocasa.gguf
```

### Terminal 1: start the server

```bash
bash eval/run_robocasa_server.sh \
    -m smolvla \
    -c weights/smolvla_robocasa.gguf \
    -a tcp://127.0.0.1:5555 \
    -o outputs/robocasa/_server_logs \
    -B
```

`-B` uses the existing binary. Omit it to rebuild the server target in the
already configured build directory. Wait for the server's `ready` message before
starting the client, and leave the server terminal running.

### Terminal 2: run a episode

With the same server running:

```bash
bash eval/run_robocasa_client.sh \
    --arch smolvla \
    --vla-addr tcp://127.0.0.1:5555 \
    --task CloseFridge \
    --task-prompt task_id \
    --split target \
    --n-episodes 1 \
    --seed 42 \
    --n-action-steps 50 \
    --output-dir outputs/robocasa
```

Without `--max-episode-steps`, the runner uses the registered task horizon:
900 steps for `CloseFridge` in the pinned version. Success ends an episode early.
SmolVLA executes 50 actions per prediction by default; use `--n-action-steps`
from 1 to 50 to change replay length.

The shell wrapper runs this Python entrypoint; it can also be called directly:

```bash
eval/sim/robocasa/robocasa_uv/.venv/bin/python \
    eval/client/run_robocasa_client_direct.py \
    --arch smolvla --task CloseFridge --n-episodes 1 \
    --vla-addr tcp://127.0.0.1:5555 \
    --output-dir outputs/robocasa
```

### Checkpoint metadata and observation format

The client reads `config.json` and `policy_preprocessor.json` from
`lerobot/smolvla_robocasa` at revision
`f7af588d11e30b0069f200792fb285904b74c80b`. It fetches metadata only; GGUF
weights stay on the server. The default tokenizer is
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`; first use can download its files.

For offline operation, add both options to the client command, using existing
local files:

```text
--policy-path weights/smolvla_robocasa --tokenizer /path/to/local/tokenizer
```

The policy directory must contain both metadata JSON files. Normalization
statistics are embedded in the SmolVLA GGUF; no `--stats-json` is needed.

| Input | Checkpoint format before transport preprocessing |
|---|---|
| `observation.images.camera1` | Left agent view, CHW float32 RGB, `(3, 256, 256)`, values in `[0, 1]` |
| `observation.images.camera2` | Right agent view, same shape and range |
| `observation.images.camera3` | Wrist view, same shape and range |
| `observation.state` | Raw float32 `(16,)`: base position (3), base quaternion (4), relative EEF position (3), relative EEF quaternion (4), gripper positions (2) |
| `task` | CamelCase task ID with a trailing newline, such as `CloseFridge\n` |

The shared client resizes/pads images to 512 pixels and pads state to 32 values.
C++ applies state/action normalization using the GGUF statistics. Camera images
retain the upstream wrapper's upright orientation.

Keep the default `--task-prompt task_id` for the released SmolVLA RoboCasa
checkpoint, whose training labels use task names. For a checkpoint trained on
English episode instructions, choose `--task-prompt language`. The client prints
the selected prompt and records it in `episodes.csv`.

## 3. Run GR00T N1.6

The server needs an N1.6 GGUF whose checkpoint includes the
`robocasa_panda_omron` embodiment; the `nvidia/GR00T-N1.6-3B` base checkpoint
does. The client reads three artifacts from `weights/GR00T-N1.6` by default
(`--policy-path` changes the directory):

- `statistics.json` containing the `robocasa_panda_omron` embodiment.
- `processor_config.json` from the same checkpoint.
- `tokenizer/`, the Eagle tokenizer with its special tokens and chat template.

Setup does not download or convert GR00T weights. Stop the SmolVLA server with
**Ctrl+C** before reusing port 5555.

### Download and convert the GR00T N1.6 checkpoint

```bash
# Download the checkpoint from Hugging Face
huggingface-cli download nvidia/GR00T-N1.6-3B --local-dir weights/GR00T-N1.6

# Fetch the Eagle tokenizer into weights/GR00T-N1.6/tokenizer (no weights)
python3 eval/sim/robocasa/download_gr00t_tokenizer.py

# Convert to GGUF
python scripts/convert_gr00t_n1_6_to_gguf.py \
    --ckpt weights/GR00T-N1.6 \
    --out weights/gr00tn1d6-robocasa.gguf
```

### Terminal 1: start the GR00T server

```bash
bash eval/run_robocasa_server.sh \
    -m gr00t_n1_6 \
    -c weights/gr00tn1d6-robocasa.gguf \
    -a tcp://127.0.0.1:5555 \
    -o outputs/robocasa/_server_logs \
    -B
```

The script selects `VLA_GR00T_EMBODIMENT=robocasa_panda_omron` automatically and
exits if the GGUF has no such embodiment. Wait for `ready` before starting the
client.

### Terminal 2: run the GR00T client

```bash
bash eval/run_robocasa_client.sh \
    --arch gr00t_n1_6 \
    --vla-addr tcp://127.0.0.1:5555 \
    --task OpenDrawer \
    --split target \
    --seed 42 \
    --n-episodes 1 \
    --n-action-steps 8 \
    --output-dir outputs/robocasa
```

`--image-size` must match the GGUF vision tower: use 224 for the converter's
default, or 252 if converted with `--vision-size 252`. GR00T supports replay
lengths from 1 to 16, defaulting to 8. Add `--max-episode-steps 10` for a short
integration check and choose a separate output directory.

GR00T defaults to `--task-prompt language`. Its parser reads camera/state/action
order and normalization settings from the matching checkpoint metadata, and
uses the 16 valid actions in the padded server response. It converts the result
into named simulator action fields. Unsupported modality sets and relative/EEF
action encodings fail during parser setup.

Both models use plain `VlaCppClient` instances from
[`eval/client/vla_cpp_client.py`](../../client/vla_cpp_client.py).
[`run_robocasa_client_direct.py`](../../client/run_robocasa_client_direct.py)
connects each client to a `SmolVLARoboCasaParser` or `GR00TN16RoboCasaParser`
through `RoboCasaSimAdapter`, following the LIBERO parser pattern. Parsers live
in [`utils/sim_adapters/robocasa.py`](../../utils/sim_adapters/robocasa.py).

## 4. Read results

Each task writes its results under `<output-dir>/<arch>/<task>/`; `summary.txt`
holds the success rate and average step time.

| Location under the output directory | Contents |
|---|---|
| `<arch>/<task>/summary.json` | Episode counts, success, prompt mode, replay length, horizon, timings, and package versions |
| `<arch>/<task>/summary.txt` | Text summary |
| `<arch>/<task>/episodes.csv` | Episode seeds, prompts, steps, success, abort status, and errors |
| `<arch>/<task>/episode_*.mp4` | Left, right, and wrist views side by side |
| `_server_logs/<arch>.log` | Server model loading and inference timing logs |
| `_server_logs/<arch>.mem.json` | Server memory samples |

The `_server_logs/` rows assume the server's `-o` is `<output-dir>/_server_logs`,
as in the commands above.

A completed episode with `success=0` means the task was not solved within its
horizon; the client can still exit successfully. A simulator or inference error
records an aborted attempt and exits with code 1. Aborted attempts count as
failures, and `summary.json` marks a partial run with `"complete": false`.

Use a new output directory to preserve another run. Reusing paths replaces
previous summaries and videos. Server metrics cover that server log's lifetime;
restart with a new log directory for a separate evaluation run.

Useful client options:

| Option | Effect |
|---|---|
| `--max-episode-steps 100` | Override the horizon for every selected task |
| `--n-action-steps 1` | Request a prediction at each simulator step |
| `--view-mode single-view` | Record only the left camera; policy inputs still contain three views |
| `--no-video` | Disable video recording |
| `--fps 20` | Set video FPS without changing the controller rate |
| `--recv-timeout-ms 240000` | Allow up to four minutes per server response |

## 5. Run inference on another machine

On the inference host, bind to all interfaces:

```bash
bash eval/run_robocasa_server.sh \
    -m smolvla \
    -c weights/smolvla_robocasa.gguf \
    -a 'tcp://*:5555' \
    -o outputs/robocasa_remote/_server_logs \
    -B
```

On the simulator host, use the inference host's reachable address:

```bash
bash eval/run_robocasa_client.sh \
    --arch smolvla \
    --vla-addr tcp://192.168.1.6:5555 \
    --task CloseFridge \
    --n-episodes 1 \
    --output-dir outputs/robocasa_remote
```

The simulator host needs its own RoboCasa environment, assets, and tokenizer.
For GR00T, also keep matching statistics and processor configuration there.
GGUF weights stay on the server. Copy server logs and memory JSON into the
client output root's `_server_logs/` directory to keep a run's results together.

## 6. Valid task names

Select tasks with exact RoboCasa class names. Multiple names are comma-separated:

```bash
bash eval/run_robocasa_client.sh \
    --arch smolvla \
    --task CloseFridge,OpenDrawer \
    --split target \
    --n-episodes 5 \
    --seed 42 \
    --output-dir outputs/robocasa_selected_tasks
```

`--n-episodes` applies to each task. Seeds are `seed + episode_index` within
each task, and the action queue is cleared before every episode. The supported
scene splits are `target` (default), `pretrain`, and `all`.

The pinned RoboCasa 1.0.1 checkout (`3443cdf3dd20f9ae8ed41adba442059b506891d0`)
registers **365 benchmark task classes: 65 atomic and 300 composite**.
Pass names exactly as listed to `--task`; separate multiple names with commas.
The lists below come from the installed environment classes and
[dataset registry](robocasa/robocasa/utils/dataset_registry.py).

### Target benchmark subset — 50 tasks

These are the upstream target benchmark groups. This runner accepts their
individual task names; it does not expand group names such as `atomic_seen`.
Use `--split target` when evaluating these target-scene tasks.

**Atomic seen — 18 tasks**

```text
CloseBlenderLid
CloseFridge
CloseToasterOvenDoor
CoffeeSetupMug
NavigateKitchen
OpenCabinet
OpenDrawer
OpenStandMixerHead
PickPlaceCounterToCabinet
PickPlaceCounterToStove
PickPlaceDrawerToCounter
PickPlaceSinkToCounter
PickPlaceToasterToCounter
SlideDishwasherRack
TurnOffStove
TurnOnElectricKettle
TurnOnMicrowave
TurnOnSinkFaucet
```

**Composite seen — 16 tasks**

```text
DeliverStraw
GetToastedBread
KettleBoiling
LoadDishwasher
PackIdenticalLunches
PreSoakPan
PrepareCoffee
RinseSinkBasin
ScrubCuttingBoard
SearingMeat
SetUpCuttingStation
StackBowlsCabinet
SteamInMicrowave
StirVegetables
StoreLeftoversInBowl
WashLettuce
```

**Composite unseen — 16 tasks**

```text
ArrangeBreadBasket
ArrangeTea
BreadSelection
CategorizeCondiments
CuttingToolSelection
GarnishPancake
GatherTableware
HeatKebabSandwich
MakeIceLemonade
PanTransfer
PortionHotDogs
RecycleBottlesByType
SeparateFreezerRack
WaffleReheat
WashFruitColander
WeighIngredients
```

### Full task catalogue — 365 tasks

The 65 atomic tasks and 252 composite tasks in the first two lists have
registered default horizons. The remaining 48 composite tasks require an
explicit `--max-episode-steps` value in this runner.

<details>
<summary>Atomic tasks — 65</summary>

```text
AdjustToasterOvenTemperature
AdjustWaterTemperature
CheesyBread
CloseBlenderLid
CloseCabinet
CloseDishwasher
CloseDrawer
CloseElectricKettleLid
CloseFridge
CloseFridgeDrawer
CloseMicrowave
CloseOven
CloseStandMixerHead
CloseToasterOvenDoor
CoffeeServeMug
CoffeeSetupMug
LowerHeat
MakeIcedCoffee
NavigateKitchen
OpenBlenderLid
OpenCabinet
OpenDishwasher
OpenDrawer
OpenElectricKettleLid
OpenFridge
OpenFridgeDrawer
OpenMicrowave
OpenOven
OpenStandMixerHead
OpenToasterOvenDoor
PackDessert
PickPlaceCabinetToCounter
PickPlaceCounterToBlender
PickPlaceCounterToCabinet
PickPlaceCounterToDrawer
PickPlaceCounterToMicrowave
PickPlaceCounterToOven
PickPlaceCounterToSink
PickPlaceCounterToStandMixer
PickPlaceCounterToStove
PickPlaceCounterToToasterOven
PickPlaceDrawerToCounter
PickPlaceFridgeDrawerToShelf
PickPlaceFridgeShelfToDrawer
PickPlaceMicrowaveToCounter
PickPlaceSinkToCounter
PickPlaceStoveToCounter
PickPlaceToasterOvenToCounter
PickPlaceToasterToCounter
PreheatOven
SlideDishwasherRack
SlideOvenRack
SlideToasterOvenRack
StartCoffeeMachine
TurnOffMicrowave
TurnOffSinkFaucet
TurnOffStove
TurnOnBlender
TurnOnElectricKettle
TurnOnMicrowave
TurnOnSinkFaucet
TurnOnStove
TurnOnToaster
TurnOnToasterOven
TurnSinkSpout
```

</details>

<details>
<summary>Composite tasks with default horizons — 252</summary>

```text
AddIceCubes
AddLemonToFish
AddMarshmallow
AddSugarCubes
AddSweetener
AdjustHeat
AfterwashSorting
AirDryFruit
AlcoholServingPrep
AlignSilverware
ArrangeBreadBasket
ArrangeBreadBowl
ArrangeBuffetDessert
ArrangeDrinkware
ArrangeTea
ArrangeTeaAccompaniments
ArrangeUtensilsByType
ArrangeVegetables
AssembleCookingArray
BalancedMealPrep
BeverageOrganization
BeverageSorting
BlendIngredients
BlendMarinade
BowlAndCup
BreadAndCheese
BreadSelection
BreadSetupSlicing
BuildAppetizerPlate
ButterOnPan
CandleCleanup
CategorizeCondiments
CerealAndBowl
ChooseMeasuringCup
ChooseRipeFruit
CleanBoard
CleanMicrowave
ClearClutter
ClearCuttingBoard
ClearFoodWaste
ClearFreezer
ClearReceptaclesForCleaning
ClearSink
ClearSinkArea
ClearSinkSpace
CollectWashingSupplies
ColorfulSalsa
CondimentCollection
CookieDoughPrep
CoolBakedCake
CoolKettle
CreateChildFriendlyFridge
CupcakeCleanup
CutBuffetPizza
CuttingToolSelection
DateNight
DefrostByCategory
DeliverBrewedCoffee
DeliverStraw
DessertAssembly
DessertUpgrade
DisplayMeatVariety
DistributeChicken
DivideBasins
DivideBuffetTrays
DrainVeggies
DrinkwareConsolidation
DryDishes
DryDrinkware
DumpLeftovers
FillBlenderJug
FillKettle
FilterMicrowavableItem
FoodCleanup
FreezeBottledWaters
FreezeCookedFood
FreezeIceTray
FreshProduceOrganization
FryingPanAdjustment
GarnishCupcake
GarnishPancake
GatherCuttingTools
GatherMarinadeIngredients
GatherTableware
GatherVegetables
GetToastedBread
HeatKebabSandwich
HeatMug
HeatMultipleWater
HotDogSetup
JuiceFruitReamer
KettleBoiling
LemonSeasoningFish
LineUpCondiments
LoadCondimentsInFridge
LoadDishwasher
LoadFridgeByType
LoadFridgeFifo
LoadPreparedFood
MakeFruitBowl
MakeIceLemonade
MakeLoadedPotato
MatchCupAndDrink
MaximizeFreezerSpace
MealPrepStaging
MeatSkewerAssembly
MeatTransfer
MicrowaveCorrectMeal
MicrowaveDefrostMeat
MicrowaveThawing
MicrowaveThawingFridge
MixCakeFrosting
MixedFruitPlatter
MoveFreezerToFridge
MoveFridgeToFreezer
MoveToCounter
MoveToFreezerDrawer
MultistepSteaming
OrganizeBakingIngredients
OrganizeCleaningSupplies
OrganizeCoffeeCondiments
OrganizeCondiments
OrganizeMetallicUtensils
OrganizeMugsByHandle
OrganizeVegetables
OvenBroilFish
PackFoodByTemp
PackFruitContainer
PackIdenticalLunches
PanTransfer
PastryDisplay
PlaceBeveragesTogether
PlaceDishesBySink
PlaceEqualIceCubes
PlaceFoodInBowls
PlaceIceInCup
PlaceMeatInMarinade
PlaceMicrowaveSafeItem
PlaceOnDishRack
PlaceStraw
PlaceVegetablesEvenly
PlaceVeggiesInDrawer
PlateSteakMeal
PlateStoreDinner
PortionFruitBowl
PortionHotDogs
PortionInTupperware
PortionOnSize
PortionYogurt
PreRinseStation
PreSoakPan
PreheatPot
PrepForSanitizing
PrepForTenderizing
PrepFridgeForCleaning
PrepMarinatingMeat
PrepSinkForCleaning
PrepareBroilingStation
PrepareCheeseStation
PrepareCocktailStation
PrepareCoffee
PrepareDishwasher
PrepareDrinkStation
PrepareSausageCheese
PrepareSmoothie
PrepareSoupServing
PrepareStoringLeftovers
PrepareToast
PressChicken
PrewashFoodAssembly
PrewashFoodSorting
QuickThaw
RearrangeFridgeItems
RecycleBottlesBySize
RecycleBottlesByType
RecycleSodaCans
RecycleStackedYogurt
RefillCondimentStation
ReheatMeal
RemoveBroiledFish
RemoveCuttingBoardItems
ReorganizeFrozenVegetables
ResetCabinetDoors
RestockBowls
RestockCannedFood
RestockPantry
RestockSinkSupplies
RetrieveIceTray
RetrieveMeat
ReturnHeatedFood
ReturnWashingSupplies
RinseBowls
RinseCuttingBoard
RinseSinkBasin
RotatePan
SanitizePrepCuttingBoard
ScalePortioning
ScrubBowl
ScrubCuttingBoard
SearingMeat
SeasoningSpiceSetup
SeasoningSteak
SeparateFreezerRack
ServeSteak
ServeTea
ServeWarmCroissant
SetBowlsForSoup
SetUpCuttingStation
SetUpSpiceStation
SetupBowls
SetupButterPlate
SetupFruitBowl
SetupFrying
SetupSodaBowl
SetupWineGlasses
ShakePan
SimmeringSauce
SizeSorting
SnackSorting
SoakSponge
SortingCleanup
SpicyMarinade
StackBowlsCabinet
StackBowlsInSink
StackCans
StartElectricKettle
SteamInMicrowave
StirVegetables
StockingBreakfastFoods
StoreDumplings
StoreLeftoversByType
StoreLeftoversInBowl
StrainerSetup
SweetSavoryToastSetup
SweetenCoffee
SweetenHotChocolate
ThawInSink
TiltPan
ToastBagel
ToastBaguette
ToastOnCorrectRack
TongBuffetSetup
TransportCookware
TurnOffSimmeredSauceHeat
VeggieDipPrep
WaffleReheat
WarmCroissant
WashFish
WashFruitColander
WashLettuce
WeighIngredients
YogurtDelightPrep
```

</details>

<details>
<summary>Additional composite tasks requiring an explicit horizon — 48</summary>

```text
AddToSoupPot
ArrangeCuttingFruits
ArrangeSinkSanitization
BeginSlowCooking
BlendSalsaMix
BlendVegetableSauce
BoilCorn
BoilEggs
BoilPot
CheeseMixing
CleanBlenderJug
ClusterItemsForClearing
ClusterUtensilsInDrawer
CoolBakedCookies
CountertopCleanup
DistributeSteakOnPans
DrawerUtensilSort
EmptyDishRack
GarnishCake
GatherProduceWashing
MakeBananaMilkshake
MakeCheesecakeFilling
MakeChocolateMilk
OrganizeMeasuringCups
PJSandwichPrep
PackSnack
PlaceBreakfastItemsAway
PlaceLidToBoil
PrepareSandwichStation
PrepareVegetableRoasting
PrepareVeggieDip
PrepareVeggiesForSteaming
ReheatMeatOnStove
RemoveSteamedVegetables
RinseFragileItem
SanitizeSink
SeparateRawIngredients
ServeMealJuice
SortBreakfastIngredients
SteamFish
SteamVeggiesWithWater
StopSlowCooking
ToastHeatableIngredients
ToastOneSlotPair
ToasterOvenBroilFish
UtensilShuffle
WashInSaucepan
WipeTable
```

</details>

For example, run a task from the final list with an explicit step budget:

```bash
bash eval/run_robocasa_client.sh \
    --arch smolvla \
    --task BoilCorn \
    --split target \
    --n-episodes 1 \
    --max-episode-steps 900 \
    --output-dir outputs/robocasa_boil_corn
```

The 900-step limit above is an example budget, not a registered horizon for
`BoilCorn`. To inspect an extra task without a policy, use `--check-env` with
the task name and an explicit step limit.

## 7. Troubleshooting

| Symptom | Action |
|---|---|
| RoboCasa Python is missing | Run `bash eval/sim/robocasa/setup_robocasa.sh` |
| A mesh, fixture, or texture is missing | Rerun setup without `--skip-assets` and inspect the asset-download output |
| EGL initialization fails | Verify graphics-device access and the installed driver; use OSMesa if its software renderer is installed |
| `vla-server` is missing | Build the server target or set `SERVER_BIN` to an existing executable |
| Server exits before `ready` | Read its log under the server's `-o` directory and check the GGUF path |
| Port 5555 is occupied | Stop the old server, or use server `-a 5566` and client `--vla-addr tcp://127.0.0.1:5566` |
| Client waits or times out | Check server readiness, network address, and that the served model matches `--arch` |
| A task fails while resolving its horizon | For a task in the 48-task list above, pass `--max-episode-steps` explicitly |
| GR00T metadata/tokenizer error | Check that GGUF, Eagle tokenizer, statistics, processor configuration, and image size match |

Optional `robosuite_models`, Mink, and MimicGen warnings can appear in the
PandaOmron environment; those extensions were not required by the validated
`CloseFridge` rollout. `uv pip check` can report RoboCasa's unused learning-stack
requirements (`lerobot` and `tianshou`) as missing; this client uses the simulator
API directly.

Inspect all CLI options:

```bash
bash eval/sim/robocasa/setup_robocasa.sh --help
bash eval/run_robocasa_server.sh -h
bash eval/run_robocasa_client.sh --help
```

Press **Ctrl+C** in the server terminal when finished. The wrapper cleans up
its inference process, log tail, and memory sampler.
