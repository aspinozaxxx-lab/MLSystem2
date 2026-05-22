import React from "react";
import { ProjectCard } from "components";

function ProjectList({ projects }) {
  const renderCard = (props) => <ProjectCard key={props.id} {...props} />;
  return <div className="project-list">{projects.map(renderCard)}</div>;
}

export default React.memo(ProjectList);
