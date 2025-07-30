domain: "[h0] -> { S1[i0, i1] : exists (e0 = floor((32 + 32h0 - i0)/64): 64e0 <= 31 + 32h0 - i0 and 64e0 >= -31 + 32h0 - i0 and i0 >= 0 and i0 <= 999 and i0 >= -2015 + 32h0 and 32e0 >= -999 + 32h0 - i0 and i1 >= 0 and i1 <= 999 and i0 <= 32 + 32h0) }"
child:
  context: "[h0] -> { [] : h0 <= 93 and h0 >= 0 }"
  child:
    schedule: "[h0] -> [{ S1[i0, i1] -> [(i0)] }, { S1[i0, i1] -> [(i1)] }]"
    options: "[h0] -> { separate[i0] }"
