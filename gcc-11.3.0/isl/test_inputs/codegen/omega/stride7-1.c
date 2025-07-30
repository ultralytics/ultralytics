{
  for (int c0 = 1; c0 <= 3; c0 += 1)
    for (int c1 = 1; c1 <= 9; c1 += 1)
      s1(c1, c0);
  for (int c0 = 4; c0 <= 9; c0 += 1)
    for (int c1 = 1; c1 <= 9; c1 += 1) {
      if (c0 % 4 == 0)
        s0(c1, c0 / 4);
      s1(c1, c0);
    }
  for (int c0 = 3; c0 <= 9; c0 += 1)
    for (int c1 = 1; c1 <= 9; c1 += 1)
      s0(c1, c0);
}
