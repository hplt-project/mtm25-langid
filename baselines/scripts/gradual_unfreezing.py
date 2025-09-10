from transformers import TrainerCallback


class GradualUnfreezingCallback(TrainerCallback):

    def __init__(self, model, unfreezing_schedule):
        self.model = model
        self.unfreezing_schedule = sorted(unfreezing_schedule.items())
        self.current_unfrozen_layers = 0

        self._freeze_all_transformer_layers()

    def _freeze_all_transformer_layers(self):
        for param in self.model.base_model.parameters():
            param.requires_grad = False
        print("All transformer layers frozen. Only classification head is trainable.")

    def _unfreeze_layers(self, num_layers):
        layers = self.model.base_model.encoder.layer

        for i in range(len(layers) - num_layers, len(layers)):
            for param in layers[i].parameters():
                param.requires_grad = True

        self.current_unfrozen_layers = num_layers
        print(f"Unfroze top {num_layers}/{len(layers)} transformer layers")
        print(f"Trainable layers: {list(range(len(layers) - num_layers, len(layers)))}")

    def on_step_begin(self, args, state, control, **kwargs):
        examples_processed = state.global_step * args.get_total_train_batch_size()

        for examples_threshold, num_layers in self.unfreezing_schedule:
            if examples_processed >= examples_threshold and num_layers > self.current_unfrozen_layers:
                print(f"\nStep {state.global_step} ({examples_processed:,} examples): Unfreezing {num_layers} layers")
                self._unfreeze_layers(num_layers)
                break
