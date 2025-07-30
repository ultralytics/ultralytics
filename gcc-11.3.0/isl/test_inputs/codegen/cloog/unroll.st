domain: "[n] -> { S1[i] : i >= 0 and i <= 10 }"
child:
  context: "[n] -> { [] }"
  child:
    schedule: "[n] -> [{ S1[i] -> [(i)] }]"
    options: "[n] -> { unroll[i0] }"
