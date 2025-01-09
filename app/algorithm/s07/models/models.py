import torch.nn as nn
import torch
from collections import namedtuple
from typing import List, Any

# from app.algorithm.s07.utils.data_preprocessing import read_platform


############################### global Attention ######################################
import torch
import torch.nn as nn

# Encoder 정의
class Encoder(nn.Module):
    def __init__(self, input_size=5, hidden_size=128):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)  # outputs: [batch_size, seq_len, hidden_size]
        return outputs, hidden, cell


# Attention 정의 수정 (Bahdanau Attention)
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))  # 학습 가능한 파라미터로 초기화
        nn.init.normal_(self.v, mean=0, std=0.1)  # 파라미터 초기화

    def forward(self, hidden, encoder_outputs):
        """
        hidden: [batch_size, hidden_size]
        encoder_outputs: [batch_size, seq_len, hidden_size]
        """
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        # hidden을 seq_len에 맞춰 확장
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_size]

        # Energy 계산
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch_size, seq_len, hidden_size]

        # v와 energy를 사용해 Attention score 계산
        # self.v를 [hidden_size] → [batch_size, hidden_size, 1]로 확장
        v = self.v.unsqueeze(0).expand(batch_size, -1).unsqueeze(2)  # [batch_size, hidden_size, 1]
        attention_weights = torch.bmm(energy, v).squeeze(-1)  # [batch_size, seq_len]

        # Softmax로 Attention 가중치 계산
        attention_weights = torch.softmax(attention_weights, dim=1)  # [batch_size, seq_len]
        attention_weights = attention_weights.unsqueeze(1)  # [batch_size, 1, seq_len]

        # Context 벡터 계산
        context = torch.bmm(attention_weights, encoder_outputs)  # [batch_size, 1, hidden_size]
        return context

# Decoder 정의
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=5):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(hidden_size + input_size, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Attention을 포함하여 예측 차원 조정

    def forward(self, x, hidden, cell, encoder_outputs):
        # Attention 계산
        # breakpoint()
        context = self.attention(hidden[-1], encoder_outputs)  # [batch_size, 1, hidden_size]
        
        # Decoder의 LSTM 입력 준비
        x = torch.cat((x, context), dim=2)  # [batch_size, 1, hidden_size + input_size]
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        
        # 출력 계산
        predictions = self.fc(torch.cat((outputs, context), dim=2))  # [batch_size, 1, output_size]
        return predictions, hidden, cell


# Seq2Seq 모델 정의
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=5):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size)
        self.decoder = Decoder(input_size=output_size, hidden_size=hidden_size, output_size=output_size)

    def forward(self, encoder_inputs, decoder_inputs,output_size):
        # Encoder
        encoder_outputs, hidden, cell = self.encoder(encoder_inputs)
        # Decoder 초기화
        outputs = []
        decoder_input = decoder_inputs[:, 0, :output_size].unsqueeze(1)  # 첫 번째 디코더 입력
        
        for t in range(decoder_inputs.size(1)):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs.append(output)
            teacher_forcing_ratio=0.5

            # Teacher Forcing
            if torch.rand(1).item() < teacher_forcing_ratio and t!=decoder_inputs.size(1)-1:
                decoder_input = decoder_inputs[:, t+1, :output_size].unsqueeze(1)
            else:
                decoder_input = output.detach()

        outputs = torch.cat(outputs, dim=1)  # 시간 순서대로 모든 예측을 연결
        return outputs
    
    def inference(self, encoder_inputs, decoder_inputs, decoder_length,output_size):
        """
        Inference method for predicting without teacher forcing.
        """
        # Encoder
        encoder_outputs, hidden, cell = self.encoder(encoder_inputs)
        # Decoder 초기화
        outputs = []
        print('decoder_input shape: ', decoder_inputs.shape)
        decoder_input = decoder_inputs[:,:,:output_size]  # 초기 디코더 입력 (시작 토큰)

        for t in range(decoder_length):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs.append(output)
            # 이전 스텝의 출력값을 다음 입력으로 사용
            decoder_input = output.detach()

        outputs = torch.cat(outputs, dim=1)  # 시간 축으로 연결
        return outputs[:,:,:3]
    
