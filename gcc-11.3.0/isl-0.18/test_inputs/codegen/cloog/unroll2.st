domain: "[n] -> { S1[i] : i >= n and i <= 1 + n and n <= 9 and i >= 0 }"
child:
  context: "[n] -> { [] }"
  child:
    schedule: "[n] -> [{ S1[i] -> [(i)] }]"
    options: "[n] -> { unroll[i0] }"
