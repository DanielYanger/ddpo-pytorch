# from diffusers.models import UNet1DModel
from CustomUNet1D import CustomUnet1D
from diffusers.utils.torch_utils import randn_tensor
import torch as t
from diffusers.schedulers import DDIMScheduler

import torch
from protein_generator import Protein
import math
import matplotlib.pyplot as plt

protein_seq = "CPEEWEWNRSVMSVHNLCWQQAVDLGLWWILVPMIGGMIYMRQPLHRWLASSFKVFAIYVSIGGQVKRWPVVRFYSMEVWDYLWGYNYYELCIVKCGNYEEKLNIYTDMNRANWPLQFKSWKGGFKGSQYKHAKGTQLRGVSWSRRDTGFCDTMRMRLDWKISWTKHAMIQQRRLFQCSVKFKCFAIGGKEKWWCPMGGKHRGEPLPPKNYCPMVEHYIWFWYFGLFVKRRQDNTRLQKLICLILDNFPCIDNNYDTCYTIEMPDLLCATEQNQCRDMDCYKHPREACIECEGCEPDTWGVSDNTNNKFGICFHRTPQKGLQSTEEIRGDPRGLYKTRGGLMDGWYVNAYFHFTQFHFYDWLEKCCMGIFQEYCMVHEYHANVIIGKVYRQQMCPGYYWKTAMPKFWWHIFNLPSKEITQFIKEVNQYLESQSDTKIKCEAKKGTRRLSFLNCVLLELYCDRDIQMECQRWVRKPWHNQHFSNLRFAGTYSWDQQLRYNTATAAVIKNTASVFTEWCRDLSKTPAMGRFATEAKAGNFKAWKMAHCKRVAPLKKMCQFEFQDVSNWAEFVRDWEFSHREWRAEFVNDLIPDINKLPQSSNTHISNKCYDQNQWTIMIEHAQPMDYMHTGQIKKVMSVGHGMYYPHCISQITWINSFIDTANTKDDHMPSQQRVPSTTSNEHKRYVAMFFSVVYGNTKFNWGNPGHHKPHAPLHTALQNFNTFFFAYTVPGRMHYWWHHVHYLWLPDFWCLCSMKDWCHHSQSKRYGVPLSQYEVDGCQDVWRMQKNMDTQFVLNWLDSGRAQGSACTEINPCPKVKMNSPCQNFHSRMWFRMRKPHLGVEFLIPNDGAKNFFLVDFCIFMMGCCMSRNVKPVMGTPCPHMYLSNHQTVQLIMDQNRFQERAIWYANDRQIDWLHNAVETTAYTYTTWRHEGHLDVLRADVVMWHFSWDVFYYCVQWFQIMNWFHDNGNVHLVSWYLSNAAYKEYSFFVTMQMKAPVQSIS"
protein = Protein(protein_seq)

total_error = 0

error_plot = []
individual_error = []
uniqueness = []

tensor = torch.load("Custom_UNet_Model/sample-6.png")
tensor_error = 0
error_count = 0
for seq in tensor:
    print(seq)
    error = protein.validate_sequence(seq)
    print(error)
    total_error+=error
    tensor_error+=error
    if error > 0:
        error_count += 1
error_plot.append(tensor_error)
individual_error.append(error_count/len(tensor))

# reshaped_tensor = tensor.view(len(tensor), -1)
# unique_rows, _ = torch.unique(reshaped_tensor, dim=0, return_inverse=True)
# uniqueness.append(unique_rows.size(0)/len(tensor))

print(error_plot)
print(individual_error)
# print(uniqueness)
