import React from "react";
import { Breadcrumbs as Bp3Breadcrumbs } from "@blueprintjs/core";

const BREADCRUMBS = [
  { href: "/users", icon: "folder-close", text: "Users" },
  { href: "/users/janet", icon: "folder-close", text: "Janet" },
  { icon: "document", text: "image.jpg" },
];

function Breadcrumbs() {
  return (
    <div>
      <Bp3Breadcrumbs className="breadcrumbs" items={BREADCRUMBS} />
    </div>
  );
}

export default React.memo(Breadcrumbs);
