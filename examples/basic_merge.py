import sd_mecha
sd_mecha.set_log_level()


# plan a simple weighted sum
a = sd_mecha.model("ghostmix_v20Bakedvae.safetensors")
b = sd_mecha.model("dreamshaper_332BakedVaeClipFix.safetensors")
recipe = sd_mecha.weighted_sum(a, b)

# perform the entire merge plan and save to output path
sd_mecha.merge(recipe, output="output.safetensors")
