domain: "[n] -> { S2[i0, i1, i2, i1, i0] : i2 >= 1 and i2 <= n and i2 <= -1 + i1 and i1 <= n and i2 <= -1 + i0 and i0 <= n; S1[i0, n, i0, i3] : i0 >= 1 and i0 <= n and i3 >= 1 + i0 and i3 <= n }"
child:
  context: "[n] -> { [] }"
  child:
    schedule: "[n] -> [{ S2[i0, i1, i2, i3, i4] -> [(i0)]; S1[i0, i1, i2, i3] -> [(i0)] }, { S2[i0, i1, i2, i3, i4] -> [(i1)]; S1[i0, i1, i2, i3] -> [(i1)] }, { S2[i0, i1, i2, i3, i4] -> [(i2)]; S1[i0, i1, i2, i3] -> [(i2)] }, { S2[i0, i1, i2, i3, i4] -> [(i3)]; S1[i0, i1, i2, i3] -> [(i3)] }, { S2[i0, i1, i2, i3, i4] -> [(i4)]; S1[i0, i1, i2, i3] -> [(0)] }]"
    options: "[n] -> { separate[i0] }"
    child:
      sequence:
      - filter: "[n] -> { S1[i0, i1, i2, i3] }"
      - filter: "[n] -> { S2[i0, i1, i2, i3, i4] }"
