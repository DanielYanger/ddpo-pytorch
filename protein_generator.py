from Bio.Data import CodonTable
import random
import numpy as np
import torch as t

class Protein:
    def __init__(self, sequence, codon_table = CodonTable.standard_dna_table):
        self.sequence = sequence
        self.codon_table = codon_table
        self.__construct_color_mapping()
        self.__construct_inverse_table()

    def __construct_color_mapping(self, A=1, U=2, C=3, G=4):
        self.base_mapping = {
            'A': A,
            A: 'A',
            'U': U,
            U: 'T',
            'C': C,
            C: 'C',
            'G': G,
            G: 'G'
        }

    def __construct_inverse_table(self):
        self.__inverse_table = {}
        for base1 in "ATCG":
            for base2 in "ATCG":
                for base3 in "ATCG":
                    codon = base1 + base2 + base3
                    try:
                        amino = self.codon_table.forward_table[codon]
                    except KeyError:
                        continue
                    codon = codon.replace('T', 'U')
                    try:
                        self.__inverse_table[amino].append(codon)
                    except KeyError:
                        self.__inverse_table[amino] = [codon]
        self.__inverse_table['B'] = self.__inverse_table['N'] + self.__inverse_table['D']
        self.__inverse_table['Z'] = self.__inverse_table['Q'] + self.__inverse_table['E']

    def __generate_random_sequence(self):
        seq = []
        for amino_acid in self.sequence:
            try:
                codon = random.choice(self.__inverse_table[amino_acid])
                seq.append([self.base_mapping[codon[0]], self.base_mapping[codon[1]], self.base_mapping[codon[2]]])
            except KeyError:
                print("Amino Acid not in codon table")
        return np.array(seq, dtype=float).transpose()/4.0
    
    def generate_n_sequences(self, n):
        seqs = []
        for _ in range(n):
            seqs.append(self.__generate_random_sequence())
        return t.from_numpy(np.array(seqs))
    
    def sequence_generator(self):
        yield t.from_numpy(np.array(self.__generate_random_sequence()))

    def validate_sequence(self, seq):
        seq = (seq.cpu().numpy().transpose()*4.0).round()
        mismatch = 0
        for codon, amino_acid, i in zip(seq, self.sequence, range(len(seq))):
            try:
                codon_str = self.base_mapping[codon[0]]+self.base_mapping[codon[1]]+self.base_mapping[codon[2]]
                if not self.codon_table.forward_table[codon_str] == amino_acid:
                    mismatch+=1
            except KeyError:
                mismatch+=1
        return mismatch
    
    def maximize_base(self, seq):
        # seq = (seq.cpu().numpy().transpose()*4.0).round()
        # mismatch = 0
        # count = 0
        # incorrect = False
        # for codon, amino_acid, i in zip(seq, self.sequence, range(len(seq))):
        #     try:
        #         codon_str = self.base_mapping[codon[0]]+self.base_mapping[codon[1]]+self.base_mapping[codon[2]]
        #         if not self.codon_table.forward_table[codon_str] == amino_acid:
        #             mismatch+=1
        #             incorrect = True
        #         else:
        #             count+=codon_str.count('G')
        #     except KeyError:
        #         mismatch+=1
        #         incorrect = True
        # return mismatch, count if not incorrect else 0
    
        seq = (seq.cpu().numpy()*4.0).round()
        reward = []
        for sequence in seq:
            sequence = sequence.transpose()
            count = 0
            for codon, amino_acid in zip(sequence, self.sequence):
                try:
                    codon_str = self.base_mapping[codon[0]]+self.base_mapping[codon[1]]+self.base_mapping[codon[2]]
                    if not self.codon_table.forward_table[codon_str] == amino_acid:
                        count = 0
                        break
                    count+=codon_str.count('G')
                except KeyError:
                    count = 0
                    break
            reward.append(count)
        return t.tensor(reward)


class ProteinReward:
    def __init__(self, protein: Protein, base) -> None:
        self.base = base
        self.protein = protein

    def maximize_base(self, sequence: t.Tensor) -> t.Tensor:
        seq = (sequence.cpu().numpy().transpose()*4.0).round()
        reward = []
        for sequence in seq:
            count = 0
            for codon, amino_acid in zip(sequence, self.protein.sequence):
                try:
                    codon_str = self.protein.base_mapping[codon[0]]+self.protein.base_mapping[codon[1]]+self.protein.base_mapping[codon[2]]
                    if not self.protein.codon_table.forward_table[codon_str] == amino_acid:
                        count = 0
                        break
                    count+=codon_str.count(self.base)
                except KeyError:
                    count = 0
                    break
            reward.append(count)

        return t.tensor(reward)