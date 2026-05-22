import React from "react";
import {
  Alignment,
  Navbar,
  NavbarGroup,
  NavbarHeading,
  NavbarDivider,
} from "@blueprintjs/core";
import { Link } from "react-router-dom";

import { MAIN } from "constants/routes";
import { SignOutButton, LanguageButton, ThemeButton } from "containers";

import { ReactComponent as Logo } from "./header-logo.svg";

function Header() {
  return (
    <div className="app-header">
      <Navbar>
        <NavbarGroup align={Alignment.LEFT}>
          <NavbarHeading>
            <Link className="logo" to={MAIN}>
              <Logo />
            </Link>
          </NavbarHeading>
        </NavbarGroup>
        <NavbarGroup align={Alignment.RIGHT}>
          <ThemeButton minimal />
          {/* <LanguageButton minimal /> */}
          <NavbarDivider />
          <SignOutButton />
        </NavbarGroup>
      </Navbar>
    </div>
  );
}

export default React.memo(Header);
