# from diffusers.pipelines import StableDiffusionPipeline
# from diffusers.models import UNet1DModel
from DDPODiffusionPipeline import DDPODiffusionPipeline1D
from CustomUNet1D import CustomUnet1D
from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from protein_generator import Protein
from diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
# pipeline = DDPODiffusionPipeline1D.from_pretrained("runwayml/stable-diffusion-v1-5", revision="main")

# pipeline.unet = CustomUnet1D(8, 1000)

# pipeline = DDPODiffusionPipeline1D(
#     CustomUnet1D(8,1000),
#     DDIMScheduler()
# )

unet = CustomUnet1D.from_pretrained("./Custom_UNet_Model/unet")
scheduler = DDIMScheduler(
            num_train_timesteps=1000, 
            clip_sample = False,
        )

# image = randn_tensor((16,3,1000))

# scheduler.set_timesteps(100)

# for i,t in enumerate(scheduler.timesteps):
#     print(i, t)
#     # 1. predict noise model_output
#     model_output = unet(image, t).sample

#     # 2. compute previous image: x_t -> x_t-1
#     image = scheduler.step(model_output, t, image).prev_sample

# image = (image / 2 + 0.5).clamp(0, 1)
# image = image.cpu().detach().numpy()
# processed = (image * 4).round().astype("uint8")

protein_seq = "CPEEWEWNRSVMSVHNLCWQQAVDLGLWWILVPMIGGMIYMRQPLHRWLASSFKVFAIYVSIGGQVKRWPVVRFYSMEVWDYLWGYNYYELCIVKCGNYEEKLNIYTDMNRANWPLQFKSWKGGFKGSQYKHAKGTQLRGVSWSRRDTGFCDTMRMRLDWKISWTKHAMIQQRRLFQCSVKFKCFAIGGKEKWWCPMGGKHRGEPLPPKNYCPMVEHYIWFWYFGLFVKRRQDNTRLQKLICLILDNFPCIDNNYDTCYTIEMPDLLCATEQNQCRDMDCYKHPREACIECEGCEPDTWGVSDNTNNKFGICFHRTPQKGLQSTEEIRGDPRGLYKTRGGLMDGWYVNAYFHFTQFHFYDWLEKCCMGIFQEYCMVHEYHANVIIGKVYRQQMCPGYYWKTAMPKFWWHIFNLPSKEITQFIKEVNQYLESQSDTKIKCEAKKGTRRLSFLNCVLLELYCDRDIQMECQRWVRKPWHNQHFSNLRFAGTYSWDQQLRYNTATAAVIKNTASVFTEWCRDLSKTPAMGRFATEAKAGNFKAWKMAHCKRVAPLKKMCQFEFQDVSNWAEFVRDWEFSHREWRAEFVNDLIPDINKLPQSSNTHISNKCYDQNQWTIMIEHAQPMDYMHTGQIKKVMSVGHGMYYPHCISQITWINSFIDTANTKDDHMPSQQRVPSTTSNEHKRYVAMFFSVVYGNTKFNWGNPGHHKPHAPLHTALQNFNTFFFAYTVPGRMHYWWHHVHYLWLPDFWCLCSMKDWCHHSQSKRYGVPLSQYEVDGCQDVWRMQKNMDTQFVLNWLDSGRAQGSACTEINPCPKVKMNSPCQNFHSRMWFRMRKPHLGVEFLIPNDGAKNFFLVDFCIFMMGCCMSRNVKPVMGTPCPHMYLSNHQTVQLIMDQNRFQERAIWYANDRQIDWLHNAVETTAYTYTTWRHEGHLDVLRADVVMWHFSWDVFYYCVQWFQIMNWFHDNGNVHLVSWYLSNAAYKEYSFFVTMQMKAPVQSIS"
protein = Protein(protein_seq)

from lucid_rain_unet_trainer import GaussianDiffusion1D

diffusion = GaussianDiffusion1D(
    model = unet,
    seq_length = 1000,
    timesteps = 1000,
    objective = 'pred_v'
)

images, latents, log_probs = pipeline_with_logprob(
    DDPODiffusionPipeline1D(unet, scheduler)
)



for image in images:
    print(protein.validate_sequence(image))

print(protein.maximize_base(images))