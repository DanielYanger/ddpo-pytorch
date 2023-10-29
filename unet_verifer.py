import torch
from protein_generator import Protein, ProteinReward
import math
import matplotlib.pyplot as plt

protein_seq = "CPEEWEWNRSVMSVHNLCWQQAVDLGLWWILVPMIGGMIYMRQPLHRWLASSFKVFAIYVSIGGQVKRWPVVRFYSMEVWDYLWGYNYYELCIVKCGNYEEKLNIYTDMNRANWPLQFKSWKGGFKGSQYKHAKGTQLRGVSWSRRDTGFCDTMRMRLDWKISWTKHAMIQQRRLFQCSVKFKCFAIGGKEKWWCPMGGKHRGEPLPPKNYCPMVEHYIWFWYFGLFVKRRQDNTRLQKLICLILDNFPCIDNNYDTCYTIEMPDLLCATEQNQCRDMDCYKHPREACIECEGCEPDTWGVSDNTNNKFGICFHRTPQKGLQSTEEIRGDPRGLYKTRGGLMDGWYVNAYFHFTQFHFYDWLEKCCMGIFQEYCMVHEYHANVIIGKVYRQQMCPGYYWKTAMPKFWWHIFNLPSKEITQFIKEVNQYLESQSDTKIKCEAKKGTRRLSFLNCVLLELYCDRDIQMECQRWVRKPWHNQHFSNLRFAGTYSWDQQLRYNTATAAVIKNTASVFTEWCRDLSKTPAMGRFATEAKAGNFKAWKMAHCKRVAPLKKMCQFEFQDVSNWAEFVRDWEFSHREWRAEFVNDLIPDINKLPQSSNTHISNKCYDQNQWTIMIEHAQPMDYMHTGQIKKVMSVGHGMYYPHCISQITWINSFIDTANTKDDHMPSQQRVPSTTSNEHKRYVAMFFSVVYGNTKFNWGNPGHHKPHAPLHTALQNFNTFFFAYTVPGRMHYWWHHVHYLWLPDFWCLCSMKDWCHHSQSKRYGVPLSQYEVDGCQDVWRMQKNMDTQFVLNWLDSGRAQGSACTEINPCPKVKMNSPCQNFHSRMWFRMRKPHLGVEFLIPNDGAKNFFLVDFCIFMMGCCMSRNVKPVMGTPCPHMYLSNHQTVQLIMDQNRFQERAIWYANDRQIDWLHNAVETTAYTYTTWRHEGHLDVLRADVVMWHFSWDVFYYCVQWFQIMNWFHDNGNVHLVSWYLSNAAYKEYSFFVTMQMKAPVQSIS"
protein = Protein(protein_seq)
protein_reward = ProteinReward(protein, 'G')
total_error = 0

error_plot = []
individual_error = []
uniqueness = []
reward =[]

from DDPODiffusionPipeline import DDPODiffusionPipeline1D
pipeline = DDPODiffusionPipeline1D.from_pretrained('./Custom_UNet_Model/', use_safetensors=True)

generator = torch.Generator(device=pipeline.device).manual_seed(0)

for i in range(1, 20):
    # tensor = torch.load(f"./Custom_UNet_Model/sample-{i}.png")
    tensor = pipeline(batch_size=10, generator=generator)
    tensor_error = 0
    error_count = 0
    for seq in tensor:
        error, count = protein.maximize_base(seq)
        total_error+=error
        tensor_error+=error
        if error > 0:
            error_count += 1
        print(error, count)
    error_plot.append(tensor_error)
    individual_error.append(error_count/len(tensor) * 100)

    reshaped_tensor = tensor.view(len(tensor), -1)
    unique_rows, _ = torch.unique(reshaped_tensor, dim=0, return_inverse=True)
    uniqueness.append(unique_rows.size(0)/len(tensor) * 100)

print(error_plot)
print(individual_error)
print(reward)

plt.plot(range(len(error_plot)), error_plot, marker='o', linestyle='-')
plt.xlabel('Model Checkpoint')
plt.ylabel('Error')

# Display the plot
plt.grid(True)
plt.savefig('./analysis/error.png')

plt.cla()
plt.clf()

plt.plot(range(len(individual_error)), individual_error, linestyle='-')
plt.xlabel('Model Checkpoint')
plt.ylabel('Error Percentage')
plt.title('Error Percent')

# Display the plot
plt.grid(True)
plt.savefig('./analysis/error_precent.png')

plt.cla()
plt.clf()

plt.plot(range(len(uniqueness)), uniqueness, linestyle='-')
plt.xlabel('Model Checkpoint')
plt.ylabel('Uniqueness Percentage')
plt.title('Uniqueness Percent')

# Display the plot
plt.grid(True)
plt.savefig('./analysis/unique_precent.png')

