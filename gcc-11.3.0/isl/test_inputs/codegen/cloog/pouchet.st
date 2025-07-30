domain: "[Ny] -> { S1[i0, i1, 2i0, -2i0 + 2i1, i4] : i0 >= 0 and i0 <= 1 and i1 >= 1 + i0 and 2i1 <= -1 + Ny + 2i0 and i4 >= 1 and i4 <= 2; S2[i0, i1, 2i0, -1 - 2i0 + 2i1, i4] : i0 >= 0 and i0 <= 1 and i1 >= 1 + i0 and 2i1 <= Ny + 2i0 and i4 >= 1 and i4 <= 2 }"
child:
  context: "[Ny] -> { [] }"
  child:
    schedule: "[Ny] -> [{ S1[i0, i1, i2, i3, i4] -> [(i0 + i1)]; S2[i0, i1, i2, i3, i4] -> [(i0 + i1)] }, { S1[i0, i1, i2, i3, i4] -> [(i1)]; S2[i0, i1, i2, i3, i4] -> [(i1)] }, { S1[i0, i1, i2, i3, i4] -> [(i4)]; S2[i0, i1, i2, i3, i4] -> [(i4)] }, { S1[i0, i1, i2, i3, i4] -> [(i2)]; S2[i0, i1, i2, i3, i4] -> [(i2)] }, { S1[i0, i1, i2, i3, i4] -> [(i3)]; S2[i0, i1, i2, i3, i4] -> [(1 + i3)] }, { S1[i0, i1, i2, i3, i4] -> [(i4)]; S2[i0, i1, i2, i3, i4] -> [(1 + i4)] }]"
    options: "[Ny] -> { separate[x] : x >= 2 }"
