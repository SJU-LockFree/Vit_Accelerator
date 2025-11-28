#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "Network.h"
//#include "ViT_seq.h"
#include "ViT_opencl.h"
#include "comparator.h"

#define _CRT_SECURE_NO_WARNINGS
#define MAX_SOURCE_SIZE (0x100000)

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL Error: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    }

const char* imagenet_label[1000] = {
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead", "electric ray", "stingray", "cock", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "robin", "bulbul", "jay", "magpie", "chickadee", "water ouzel", "kite", "bald eagle", "vulture", "great grey owl", "European fire salamander", "common newt", "eft", "spotted salamander", "axolotl", "bullfrog", "tree frog", "tailed frog", "loggerhead", "leatherback turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "common iguana", "American chameleon", "whiptail", "agama", "frilled lizard", "alligator lizard", "Gila monster", "green lizard", "African chameleon", "Komodo dragon", "African crocodile", "American alligator", "triceratops", "thunder snake", "ringneck snake", "hognose snake", "green snake", "king snake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "rock python", "Indian cobra", "green mamba", "sea snake", "horned viper", "diamondback", "sidewinder", "trilobite", "harvestman", "scorpion", "black and gold garden spider", "barn spider", "garden spider", "black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie chicken", "peacock", "quail", "partridge", "African grey", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "drake", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "American egret", "bittern", "crane bird", "limpkin", "European gallinule", "American coot", "bustard", "ruddy turnstone", "red-backed sandpiper", "redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese spaniel", "Maltese dog", "Pekinese", "Shih-Tzu", "Blenheim spaniel", "papillon", "toy terrier", "Rhodesian ridgeback", "Afghan hound", "basset", "beagle", "bloodhound", "bluetick", "black-and-tan coonhound", "Walker hound", "English foxhound", "redbone", "borzoi", "Irish wolfhound", "Italian greyhound", "whippet", "Ibizan hound", "Norwegian elkhound", "otterhound", "Saluki", "Scottish deerhound", "Weimaraner", "Staffordshire bullterrier", "American Staffordshire terrier", "Bedlington terrier", "Border terrier", "Kerry blue terrier", "Irish terrier", "Norfolk terrier", "Norwich terrier", "Yorkshire terrier", "wire-haired fox terrier", "Lakeland terrier", "Sealyham terrier", "Airedale", "cairn", "Australian terrier", "Dandie Dinmont", "Boston bull", "miniature schnauzer", "giant schnauzer", "standard schnauzer", "Scotch terrier", "Tibetan terrier", "silky terrier", "soft-coated wheaten terrier", "West Highland white terrier", "Lhasa", "flat-coated retriever", "curly-coated retriever", "golden retriever", "Labrador retriever", "Chesapeake Bay retriever", "German short-haired pointer", "vizsla", "English setter", "Irish setter", "Gordon setter", "Brittany spaniel", "clumber", "English springer", "Welsh springer spaniel", "cocker spaniel", "Sussex spaniel", "Irish water spaniel", "kuvasz", "schipperke", "groenendael", "malinois", "briard", "kelpie", "komondor", "Old English sheepdog", "Shetland sheepdog", "collie", "Border collie", "Bouvier des Flandres", "Rottweiler", "German shepherd", "Doberman", "miniature pinscher", "Greater Swiss Mountain dog", "Bernese mountain dog", "Appenzeller", "EntleBucher", "boxer", "bull mastiff", "Tibetan mastiff", "French bulldog", "Great Dane", "Saint Bernard", "Eskimo dog", "malamute", "Siberian husky", "dalmatian", "affenpinscher", "basenji", "pug", "Leonberg", "Newfoundland", "Great Pyrenees", "Samoyed", "Pomeranian", "chow", "keeshond", "Brabancon griffon", "Pembroke", "Cardigan", "toy poodle", "miniature poodle", "standard poodle", "Mexican hairless", "timber wolf", "white wolf", "red wolf", "coyote", "dingo", "dhole", "African hunting dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby", "tiger cat", "Persian cat", "Siamese cat", "Egyptian cat", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "ice bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "long-horned beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket", "walking stick", "cockroach", "mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "admiral", "ringlet", "monarch", "cabbage butterfly", "sulphur butterfly", "lycaenid", "starfish", "sea urchin", "sea cucumber", "wood rabbit", "hare", "Angora", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "sorrel", "zebra", "hog", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram", "bighorn", "ibex", "hartebeest", "impala", "gazelle", "Arabian camel", "llama", "weasel", "mink", "polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas", "baboon", "macaque", "langur", "colobus", "proboscis monkey", "marmoset", "capuchin", "howler monkey", "titi", "spider monkey", "squirrel monkey", "Madagascar cat", "indri", "Indian elephant", "African elephant", "lesser panda", "giant panda", "barracouta", "eel", "coho", "rock beauty", "anemone fish", "sturgeon", "gar", "lionfish", "puffer", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibian", "analog clock", "apiary", "apron", "ashcan", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint", "Band Aid", "banjo", "bannister", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "barrow", "baseball", "basketball", "bassinet", "bassoon", "bathing cap", "bath towel", "bathtub", "beach wagon", "beacon", "beaker", "bearskin", "beer bottle", "beer glass", "bell cote", "bib", "bicycle-built-for-two", "bikini", "binder", "binoculars", "birdhouse", "boathouse", "bobsled", "bolo tie", "bonnet", "bookcase", "bookshop", "bottlecap", "bow", "bow tie", "brass", "brassiere", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "bullet train", "butcher shop", "cab", "caldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "carpenter's kit", "carton", "car wheel", "cash machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "cellular telephone", "chain", "chainlink fence", "chain mail", "chain saw", "chest", "chiffonier", "chime", "china cabinet", "Christmas stocking", "church", "cinema", "cleaver", "cliff dwelling", "cloak", "clog", "cocktail shaker", "coffee mug", "coffeepot", "coil", "combination lock", "computer keyboard", "confectionery", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "crane", "crash helmet", "crate", "crib", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishrag", "dishwasher", "disk brake", "dock", "dogsled", "dome", "doormat", "drilling platform", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso maker", "face powder", "feather boa", "file", "fireboat", "fire engine", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gasmask", "gas pump", "goblet", "go-kart", "golf ball", "golfcart", "gondola", "gong", "gown", "grand piano", "greenhouse", "grille", "grocery store", "guillotine", "hair slide", "hair spray", "half track", "hammer", "hamper", "hand blower", "hand-held computer", "handkerchief", "hard disc", "harmonica", "harp", "harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoopskirt", "horizontal bar", "horse cart", "hourglass", "iPod", "iron", "jack-o'-lantern", "jean", "jeep", "jersey", "jigsaw puzzle", "jinrikisha", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "liner", "lipstick", "Loafer", "lotion", "loudspeaker", "loupe", "lumbermill", "magnetic compass", "mailbag", "mailbox", "maillot", "maillot tank suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine chest", "megalith", "microphone", "microwave", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "Model T", "modem", "monastery", "monitor", "moped", "mortar", "mortarboard", "mosque", "mosquito net", "motor scooter", "mountain bike", "mountain tent", "mouse", "mousetrap", "moving van", "muzzle", "nail", "neck brace", "necklace", "nipple", "notebook", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "organ", "oscilloscope", "overskirt", "oxcart", "oxygen mask", "packet", "paddle", "paddlewheel", "padlock", "paintbrush", "pajama", "palace", "panpipe", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "passenger car", "patio", "pay-phone", "pedestal", "pencil box", "pencil sharpener", "perfume", "Petri dish", "photocopier", "pick", "pickelhaube", "picket fence", "pickup", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate", "pitcher", "plane", "planetarium", "plastic bag", "plate rack", "plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "pop bottle", "pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "projectile", "projector", "puck", "punching bag", "purse", "quill", "quilt", "racer", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "rubber eraser", "rugby ball", "rule", "running shoe", "safe", "safety pin", "saltshaker", "sandal", "sarong", "sax", "scabbard", "scale", "school bus", "schooner", "scoreboard", "screen", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe shop", "shoji", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "ski mask", "sleeping bag", "slide rule", "sliding door", "slot", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar dish", "sombrero", "soup bowl", "space bar", "space heater", "space shuttle", "spatula", "speedboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "steel arch bridge", "steel drum", "stethoscope", "stole", "stone wall", "stopwatch", "stove", "strainer", "streetcar", "stretcher", "studio couch", "stupa", "submarine", "suit", "sundial", "sunglass", "sunglasses", "sunscreen", "suspension bridge", "swab", "sweatshirt", "swimming trunks", "swing", "switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy", "television", "tennis ball", "thatch", "theater curtain", "thimble", "thresher", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toyshop", "tractor", "trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright", "vacuum", "vase", "vault", "velvet", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "warplane", "washbasin", "washer", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "wig", "window screen", "window shade", "Windsor tie", "wine bottle", "wing", "wok", "wooden spoon", "wool", "worm fence", "wreck", "yawl", "yurt", "web site", "comic book", "crossword puzzle", "street sign", "traffic light", "book jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "ice lolly", "French loaf", "bagel", "pretzel", "cheeseburger", "hotdog", "mashed potato", "head cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "custard apple", "pomegranate", "hay", "carbonara", "chocolate sauce", "dough", "meat loaf", "pizza", "potpie", "burrito", "red wine", "espresso", "cup", "eggnog", "alp", "bubble", "cliff", "coral reef", "geyser", "lakeside", "promontory", "sandbar", "seashore", "valley", "volcano", "ballplayer", "groom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "hip", "buckeye", "coral fungus", "agaric", "gyromitra", "stinkhorn", "earthstar", "hen-of-the-woods", "bolete", "ear", "toilet tissue"
};

char* load_kernel_source(const char* filename) {
    FILE* fp;
    errno_t err = fopen_s(&fp, filename, "r");
    if (err != 0 || !fp) {
        printf("Failed to load kernel file: %s\n", filename);
        exit(1);
    }
    char* source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    source_str[source_size] = '\0';
    fclose(fp);
    return source_str;
}

int main() {
    cl_int err;

    ////////////////////////////////////// Input load //////////////////////////////////////
    // 이미지 데이터 로드
    const char* img_filename = "./Data/input-100.bin";
    ImageData* images = load_image_data(img_filename);
    if (images == NULL) {
        printf("Failed to load images.\n");
        return 1;
    }

    ////////////////////////////////////// Weight load //////////////////////////////////////
    // 가중치 로드 (Network.h의 구조체 사용)
    Network network[152];
    load_weights("./Network", network, 152);

    ////////////////////////////////////// OpenCL Setup //////////////////////////////////////
    printf("Initializing OpenCL...\n");

    // 1. 플랫폼 및 디바이스(GPU) 설정
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices, ret_num_platforms;

    err = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    CHECK_ERROR(err);

    // GPU 우선 검색, 실패 시 CPU 사용
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    if (err != CL_SUCCESS) {
        printf("Warning: GPU not found. Trying CPU...\n");
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &ret_num_devices);
    }
    CHECK_ERROR(err);

    // 2. Context & Queue 생성
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    CHECK_ERROR(err);

    // 프로파일링 기능 활성화 (시간 측정 등 필요시 사용)
    cl_command_queue_properties props = 0; // CL_QUEUE_PROFILING_ENABLE;
    cl_command_queue queue = clCreateCommandQueue(context, device_id, props, &err);
    CHECK_ERROR(err);

    // 3. 커널 소스 로드 및 빌드
    char* kernel_source = load_kernel_source("kernel.cl");
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, NULL, &err);
    CHECK_ERROR(err);

    const char* options = "-cl-fast-relaxed-math";
    err = clBuildProgram(program, 1, &device_id, options, NULL, NULL);
    if (err != CL_SUCCESS) {
        // 빌드 에러 로그 출력 (필수)
        char buffer[4096];
        size_t len;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("Build Log:\n%s\n", buffer);
        free(kernel_source);
        exit(1);
    }
    free(kernel_source);

    ////////////////////////////////////// Weight to GPU //////////////////////////////////////
    printf("Moving Weights to GPU...\n");

    // 152개의 레이어 가중치를 담을 GPU 메모리 배열
    cl_mem d_network[152];

    for (int i = 0; i < 152; ++i) {
        // Network 구조체에 이미 size 정보가 있으므로 바로 사용!
        size_t count = network[i].size;

        if (count > 0 && network[i].data != NULL) {
            // Context 생성 시 CL_MEM_COPY_HOST_PTR 플래그를 쓰면 WriteBuffer 없이 생성과 동시에 복사됨
            d_network[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(float) * count, network[i].data, &err);
            CHECK_ERROR(err);
        }
        else {
            // 가중치가 없는 레이어(예: 일부 Identity 레이어 등) 처리
            d_network[i] = NULL;
        }
    }

    ////////////////////////////////////// Inference //////////////////////////////////////
    int n = images->n; // 이미지 개수
    float** probabilities = (float**)malloc(sizeof(float*) * n);
    for (int i = 0; i < n; i++) {
        probabilities[i] = (float*)malloc(sizeof(float) * 1000);
    }

    printf("===================== Start OpenCL Inference ========================\n");
    time_t start = clock();

    // ★ 핵심: ViT_opencl 함수 호출 (OpenCL 버전)
    // 이 함수 안에서 각 레이어별 커널(kernel)을 생성하고 실행해야 함
    ViT_opencl(images, d_network, probabilities, context, queue, program);

    // 대기열의 모든 명령이 끝날 때까지 대기 (그래야 정확한 시간 측정 가능)
    clFinish(queue);

    time_t end = clock();
    printf("Elapsed time: %.2f sec\n", (double)(end - start) / CLK_TCK);

    ////////////////////////////////////// Result Save & Verify //////////////////////////////////////
    FILE* fp_output = NULL;
    errno_t open_err = fopen_s(&fp_output, "./Data/opencl_result.txt", "w");
    if (open_err != 0 || !fp_output) {
        printf("Error: cannot open output file\n");
        return 1;
    }

    // 결과 파일 작성
    for (int i = 0; i < n; i++) {
        int pred_idx = 0;
        for (int j = 1; j < 1000; j++) {
            if (probabilities[i][j] > probabilities[i][pred_idx]) {
                pred_idx = j;
            }
        }
        // comparator가 읽을 수 있는 포맷 유지
        fprintf(fp_output, "[%d] label: %d / prob: %.6f\n", i, pred_idx, probabilities[i][pred_idx]);
    }
    fclose(fp_output);

    // 정답 비교 (comparator.h)
    printf("\n[Verification]\n");
    int cmp = comparator();
    if (cmp == 0) {
        printf("PASS: OpenCL Result Matches CPU Result Perfectly!\n");
    }
    else if (cmp > 0) {
        printf("FAIL: Found %d mismatches.\n", cmp);
    }
    else {
        printf("FAIL: Comparator error.\n");
    }

    ////////////////////////////////////// Cleanup //////////////////////////////////////
    // GPU 메모리 해제
    for (int i = 0; i < 152; ++i) {
        if (d_network[i]) clReleaseMemObject(d_network[i]);
    }
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // Host 메모리 해제
    for (int i = 0; i < n; i++) free(probabilities[i]);
    free(probabilities);
    // free_image_data(images); // 필요시 주석 해제

    return 0;
}