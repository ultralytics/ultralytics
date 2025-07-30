# Check that no expressions of the form ((-n + 2147483648) % 32) are produced.
domain: "[n] -> { S_2[i] : i >= 0 and i <= -1 + n; S_1[i, j] : j >= 0 and j <= -1 + n and i >= 0 and i <= -1 + n }"
child:
  context: "[n] -> { [] : n <= 2147483647 and n >= 0 }"
  child:
    schedule: "[n] -> [{ S_1[i, j] -> [(32*floor((i)/32))]; S_2[i] -> [(32*floor((i)/32))] }, { S_1[i, j] -> [(32*floor((j)/32))]; S_2[i] -> [(32*floor((n)/32))] }]"
    permutable: 1
    options: "[n] -> { atomic[i0] : i0 >= 0 and i0 <= 1; isolate[[] -> [i0, i1]] : (exists (e0 = floor((i0)/32), e1 = floor((i1)/32): 32e0 = i0 and 32e1 = i1 and i0 >= 0 and i0 <= -32 + n and i1 >= 0 and i1 <= n)) or (exists (e0 = floor((i0)/32), e1 = floor((i1)/32): 32e0 = i0 and 32e1 = i1 and i0 >= 0 and i0 <= -32 + n and i1 >= -31 + n and i1 <= -31 + 2n)) }"
    child:
          schedule: "[n] -> [{ S_1[i, j] -> [(i - 32*floor((i)/32))]; S_2[i] -> [(i - 32*floor((i)/32))] }, { S_1[i, j] -> [(j - 32*floor((j)/32))]; S_2[i] -> [(n - 32*floor((n)/32))] }]"
          permutable: 1
          options: "{ separate[i0] : i0 >= 0 and i0 <= 1 }"
          child:
            sequence:
            - filter: "[n] -> { S_1[i, j] }"
            - filter: "[n] -> { S_2[i] }"
