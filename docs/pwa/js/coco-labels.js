export const COCO_LABELS = [
  "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
  "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
  "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
  "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
  "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
  "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
  "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
  "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
  "remote","keyboard","cell phone","microwave","oven","toaster","sink",
  "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
];

// Color palette for bounding boxes (one per class, cycled)
const PALETTE = [
  "#FF3838","#FF9D97","#FF701F","#FFB21D","#CFD231","#48F90A","#92CC17","#3DDB86",
  "#1A9334","#00D4BB","#2C99A8","#00C2FF","#344593","#6473FF","#0018EC","#8438FF",
  "#520085","#CB38FF","#FF95C8","#FF37C7"
];
export function classColor(classId) {
  return PALETTE[classId % PALETTE.length];
}

