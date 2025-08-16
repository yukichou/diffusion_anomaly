from diffusers import StableDiffusionInpaintPipeline
class StableDiffusionInpaintPipeline_dynamic(StableDiffusionInpaintPipeline):
    def __call__(self, *args, anomaly_strength=None, anomaly_stop_step=None, use_random_mask=False, **kwargs):
        if anomaly_strength is not None:
            kwargs.setdefault("strength", float(anomaly_strength))
        if anomaly_stop_step is not None:
            nis = int(kwargs.get("num_inference_steps", 50))
            kwargs["num_inference_steps"] = min(int(anomaly_stop_step), nis)
        return super().__call__(*args, **kwargs)
