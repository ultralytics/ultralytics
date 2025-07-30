for (int c0 = 4; c0 <= 100; c0 += 4) {
  for (int c1 = 1; c1 <= 100; c1 += 1)
    s0(c0, c1);
  if (c0 >= 8 && c0 <= 96)
    for (int c1 = 10; c1 <= 100; c1 += 1)
      s1(c0 + 2, c1);
}
